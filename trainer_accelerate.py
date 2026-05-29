"""
ConnecToMind2 Trainer with Accelerate (DeepSpeed ZeRO-2)

주요 변경점 (DDP -> DeepSpeed):
    1. GradScaler 제거 (Accelerator가 자동 처리)
    2. accelerator.backward() 사용
    3. accelerator.wait_for_everyone() 추가
    4. Main process만 evaluation/metric 실행
    5. Checkpoint 저장 시 unwrap_model() 사용
    6. Gradient clipping: deepspeed_config.json에서 설정 (accelerator.clip_grad_norm_ 제거)

Epoch 로직:
    - Epoch 0: train + evaluation + metric (초기 성능 확인)
    - Epoch 1~249: train only
    - Epoch 250+: train + (5 단위로 evaluation + metric)
"""

import os
import gc
from tqdm import tqdm
import wandb

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F

from utils import img_augment, versatile_diffusion_reconstruct, save_recon_batch, save_lowlevel_batch, load_recons_from_disk


# ============================================================================
# Main Training Loop
# ============================================================================

def train_evaluate_metric(args, train_loader, val_loader, test_loaders, models, optimizer, lr_scheduler, metrics, accelerator, start_epoch=0, start_global_step=0, resume_frozen_pre_qformer=False):
    """
    전체 학습 루프 (Accelerate DDP 버전)

    Epoch 로직:
        - 매 Epoch: train + validation loss 계산
        - Epoch 0: train + validation + evaluation + metric
        - Epoch 1~249: train + validation only
        - Epoch 250+: train + validation + (5 단위로 evaluation + metric)

    Args:
        start_epoch: Resume 시 시작 epoch (default: 0)
        start_global_step: Resume 시 시작 global_step (default: 0)
        resume_frozen_pre_qformer: Resume 시 frozen 상태 (default: False)
    """
    device = accelerator.device

    # ============ Models ============
    model = models["connectomind2"]  # 이미 accelerator.prepare()로 wrap됨
    versatile_diffusion = models["versatile_diffusion"]
    vae = models["vae"]

    # Output directory (main process만 생성)
    output_dir = os.path.join(args.root_dir, "5-mindeye_code/ConnecToMind2", args.output_dir)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    global_step = start_global_step
    frozen_pre_qformer = resume_frozen_pre_qformer

    # Resume 시 frozen 상태 복원
    if resume_frozen_pre_qformer:
        model_unwrapped = accelerator.unwrap_model(model)
        model_unwrapped.freeze_pre_qformer = True
        if accelerator.is_main_process:
            print(f"[Resume] Restored frozen_pre_qformer = True")

    # Curriculum Learning: FIC → FIR 전환 시점 (1/2 지점)
    fir_start_epoch = args.num_epochs * 1 // 2

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Starting Training: {args.experiment_name}")
        print(f"Total Epochs: {args.num_epochs}")
        print(f"Start Epoch: {start_epoch}")
        print(f"Batch Size (per GPU): {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * accelerator.num_processes}")
        print(f"Device: {device}")
        print(f"Curriculum: FIC (epoch 0~{fir_start_epoch-1}) → FIR (epoch {fir_start_epoch}+)")
        print(f"Frozen pre-Q-Former: {frozen_pre_qformer}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.num_epochs):

        # ============ Phase 2: Freeze pre-Q-Former modules ============
        if epoch == fir_start_epoch and not frozen_pre_qformer:
            model_unwrapped = accelerator.unwrap_model(model)
            model_unwrapped.freeze_pre_qformer = True
            frozen_pre_qformer = True

            if accelerator.is_main_process:
                print(f"\n[Epoch {epoch}] Frozen: region_embedding, bottleneck (detach mode)")

        # ============ Train ============
        if accelerator.is_main_process:
            print(f"\n[Epoch {epoch}] Training...")

        global_step, avg_loss = train_one_epoch(
            args, model, vae, train_loader,
            optimizer, lr_scheduler,
            epoch, global_step, accelerator, fir_start_epoch
        )

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # ============ Validation ============
        val_loss = validate(args, model, vae, val_loader, accelerator, epoch, fir_start_epoch)

        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")

        # WandB logging (main process only)
        if args.wandb_log and accelerator.is_main_process:
            wandb.log({"train_loss": avg_loss, "val_loss": val_loss}, step=global_step)

        # ============ Evaluation + Metric ============
        # Epoch 0 또는 Epoch 100 이상에서 10 단위
        should_evaluate = (epoch % 10 == 0) or (epoch >= 100 and epoch % 10 == 0)

        if should_evaluate:
            # 모든 프로세스가 평가 시작 전 동기화
            accelerator.wait_for_everyone()

            # 이미지 저장 디렉토리
            epoch_dir = os.path.join(output_dir, args.experiment_name, f"{args.experiment_name}_epoch{epoch}")
            accelerate_ckpt_dir = os.path.join(epoch_dir, "accelerate_checkpoint")

            # ============ Step 0: Checkpoint 저장 (inference 전에!) ============
            # 디렉토리 생성 (main만) 후 동기화
            if accelerator.is_main_process:
                os.makedirs(epoch_dir, exist_ok=True)
                print(f"\n[Epoch {epoch}] Saving checkpoint (before inference)...")
            accelerator.wait_for_everyone()  # 디렉토리 생성 완료 대기

            # DeepSpeed state 저장 (모든 process 참여 필수)
            accelerator.save_state(accelerate_ckpt_dir)

            # 메타데이터 저장 (main process만)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt_path = os.path.join(epoch_dir, f"{args.experiment_name}_epoch{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'frozen_pre_qformer': frozen_pre_qformer,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
                print(f"  Accelerate state saved: {accelerate_ckpt_dir}")

            accelerator.wait_for_everyone()

            # ============ Step 1: Inference & Metrics ============
            if accelerator.is_main_process:
                print(f"\n[Epoch {epoch}] Evaluation & Metrics (DDP across {accelerator.num_processes} GPUs)...")

            recon_dir = os.path.join(epoch_dir, "recon")
            low_recon_dir = os.path.join(epoch_dir, "low_recon")

            # ============ Step 1: 모든 Subject의 이미지 생성 및 저장 (DDP) ============
            for sub, sub_loader in test_loaders.items():
                if accelerator.is_main_process:
                    print(f"\n  [{sub}] Generating reconstructions (distributed across {accelerator.num_processes} GPUs)...")

                # 모든 GPU가 함께 evaluate 실행 (각자 다른 batch 처리)
                num_samples = evaluate(args, model, vae, versatile_diffusion, sub_loader, epoch, sub, recon_dir, low_recon_dir, accelerator)

                # 각 GPU가 처리한 샘플 수 합산
                total_samples = accelerator.gather(torch.tensor(num_samples, device=accelerator.device)).sum().item()

                if accelerator.is_main_process:
                    print(f"  [{sub}] Saved {total_samples} images to disk")

            # 모든 GPU가 이미지 생성 완료할 때까지 대기
            accelerator.wait_for_everyone()

            # ============ Step 2: 디스크에서 로드하여 Metric 계산 (Main only) ============
            all_results = {}
            if accelerator.is_main_process:
                print(f"\n  Computing metrics from saved images...")
                for sub in args.subjects:
                    print(f"\n  [{sub}] Loading images and computing metrics...")

                    # 디스크에서 이미지 로드
                    sub_recons, sub_targets = load_recons_from_disk(recon_dir, sub)

                    # Metric 계산
                    sub_results = compute_metrics(args, sub_recons, sub_targets, metrics)
                    all_results[sub] = sub_results

                    # Subject별 결과 출력
                    print(f"  [{sub}] Results:")
                    for name, score in sub_results.items():
                        print(f"    {name:15}: {score:.4f}")

                    # 메모리 정리
                    del sub_recons, sub_targets
                    torch.cuda.empty_cache()
                    gc.collect()

                # Subject 평균 계산
                results = {}
                metric_names = list(all_results[args.subjects[0]].keys())
                for name in metric_names:
                    scores = [all_results[sub][name] for sub in args.subjects]
                    results[name] = np.mean(scores)

                # 평균 결과 출력
                print(f"\n{'='*40}")
                print(f"Epoch {epoch} Average Metrics (across {len(args.subjects)} subjects):")
                print(f"{'='*40}")
                for name, score in results.items():
                    print(f"  {name:15}: {score:.4f}")
                print(f"{'='*40}\n")

                # WandB logging (평균 + subject별)
                if args.wandb_log:
                    wandb.log({f"eval/{k}": v for k, v in results.items()}, step=global_step)
                    for sub in args.subjects:
                        wandb.log({f"eval/{sub}/{k}": v for k, v in all_results[sub].items()}, step=global_step)

                # Metric 결과 텍스트 저장
                metric_path = os.path.join(epoch_dir, f"{args.experiment_name}_metric.txt")
                with open(metric_path, "w") as f:
                    f.write(f"Epoch: {epoch}\n")
                    f.write(f"{'='*40}\n")
                    f.write("Average Metrics:\n")
                    for name, score in results.items():
                        f.write(f"  {name}: {score:.4f}\n")
                    f.write(f"\n{'='*40}\n")
                    f.write("Per-Subject Metrics:\n")
                    for sub in args.subjects:
                        f.write(f"\n  [{sub}]\n")
                        for name, score in all_results[sub].items():
                            f.write(f"    {name}: {score:.4f}\n")
                print(f"Metrics saved: {metric_path}")
                print(f"Reconstructions saved: {recon_dir}")
                print(f"Low-level images saved: {low_recon_dir}")

                # 메모리 정리
                del all_results
            torch.cuda.empty_cache()
            gc.collect()

        # 모든 프로세스 동기화 (evaluation 끝날 때까지 대기)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}\n")

    return model


def evaluate_only(args, test_loaders, models, metrics, accelerator, checkpoint_epoch=None, resume_frozen_pre_qformer=False):
    """
    Evaluation-only mode: test dataset에 대해 재현 및 Metric 계산
    """
    device = accelerator.device
    model = models["connectomind2"]
    versatile_diffusion = models["versatile_diffusion"]
    vae = models["vae"]

    output_dir = os.path.join(args.root_dir, "5-mindeye_code/ConnecToMind2", args.output_dir)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    if resume_frozen_pre_qformer:
        model_unwrapped = accelerator.unwrap_model(model)
        model_unwrapped.freeze_pre_qformer = True
        if accelerator.is_main_process:
            print(f"[Resume] Restored frozen_pre_qformer = True")

    epoch_label = f"epoch{checkpoint_epoch}" if checkpoint_epoch is not None else "evaluate"
    eval_dir = os.path.join(output_dir, args.experiment_name, f"{args.experiment_name}_{epoch_label}")
    recon_dir = os.path.join(eval_dir, "recon")
    low_recon_dir = os.path.join(eval_dir, "low_recon")

    if accelerator.is_main_process:
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(low_recon_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Starting Evaluation-only: {args.experiment_name}")
        print(f"Output dir: {eval_dir}")
        print(f"Subjects: {args.subjects}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()

    for sub, sub_loader in test_loaders.items():
        if accelerator.is_main_process:
            print(f"\n  [{sub}] Generating reconstructions (distributed across {accelerator.num_processes} GPUs)...")

        num_samples = evaluate(args, model, vae, versatile_diffusion, sub_loader, epoch_label, sub, recon_dir, low_recon_dir, accelerator)
        total_samples = accelerator.gather(torch.tensor(num_samples, device=accelerator.device)).sum().item()

        if accelerator.is_main_process:
            print(f"  [{sub}] Saved {total_samples} images to disk")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n  Computing metrics from saved images...")
        all_results = {}
        for sub in args.subjects:
            print(f"\n  [{sub}] Loading images and computing metrics...")
            sub_recons, sub_targets = load_recons_from_disk(recon_dir, sub)
            sub_results = compute_metrics(args, sub_recons, sub_targets, metrics)
            all_results[sub] = sub_results

            print(f"  [{sub}] Results:")
            for name, score in sub_results.items():
                print(f"    {name:15}: {score:.4f}")

            del sub_recons, sub_targets
            torch.cuda.empty_cache()
            gc.collect()

        results = {}
        metric_names = list(all_results[args.subjects[0]].keys()) if args.subjects else []
        for name in metric_names:
            scores = [all_results[sub][name] for sub in args.subjects]
            results[name] = np.mean(scores)

        print(f"\n{'='*40}")
        print(f"Evaluation Summary ({len(args.subjects)} subjects):")
        print(f"{'='*40}")
        for name, score in results.items():
            print(f"  {name:15}: {score:.4f}")
        print(f"{'='*40}\n")

        metric_path = os.path.join(eval_dir, f"{args.experiment_name}_metric.txt")
        with open(metric_path, "w") as f:
            f.write(f"Evaluation Summary\n")
            f.write(f"Epoch label: {epoch_label}\n")
            f.write(f"{'='*40}\n")
            f.write("Average Metrics:\n")
            for name, score in results.items():
                f.write(f"  {name}: {score:.4f}\n")
            f.write(f"\n{'='*40}\n")
            f.write("Per-Subject Metrics:\n")
            for sub in args.subjects:
                f.write(f"\n  [{sub}]\n")
                for name, score in all_results[sub].items():
                    f.write(f"    {name}: {score:.4f}\n")

        print(f"Metrics saved: {metric_path}")
        print(f"Reconstructions saved: {recon_dir}")
        print(f"Low-level images saved: {low_recon_dir}")

    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Train Function
# ============================================================================

def train_one_epoch(args, model, vae, train_loader, optimizer, lr_scheduler, epoch, global_step, accelerator, fir_start_epoch):
    """
    한 epoch 학습 (Accelerate DDP 버전)

    주요 변경:
        - GradScaler 제거 (Accelerator가 자동 처리)
        - accelerator.backward() 사용
        - autocast() 제거 (Accelerator가 자동 처리)
    Curriculum: epoch < fir_start_epoch이면 FIC, 이후 FIR
    """
    device = accelerator.device
    model.train()

    losses_log = []
    lrs_log = []

    # tqdm은 main process만 표시
    if accelerator.is_main_process:
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch}", ncols=150)
    else:
        progress_bar = enumerate(train_loader)

    for batch_idx, (fmri, image, subject_name) in progress_bar:
        # Global step 계산
        current_step = global_step + batch_idx

        # Gradient 초기화
        optimizer.zero_grad()

        # Data -> GPU with correct dtype (DeepSpeed bf16 사용 시 자동 변환)
        fmri = fmri.to(device=device, dtype=torch.float16)
        image = image.to(device=device, dtype=torch.float16)

        # ============ Forward ============
        # Accelerator가 mixed precision 자동 처리
        # accelerator 전달: FIC/FIM에서 all_gather 사용 (BLIP-2 공식 방식)
        outputs = model(fmri, image, device, subject_names=subject_name, accelerator=accelerator)

        # Q-Former losses (FIR + FIC + FIM)
        loss_fir = outputs["loss_fir"]
        loss_fic = outputs["loss_fic"]
        loss_fim = outputs["loss_fim"]

        # ============ Low-level L1 Loss ============
        # VAE encode target (VAE는 float32이므로 변환 필요)
        # lowlevel_l1이 [B, 4, 32, 32] (256x256 기준)이므로 image도 256로 resize
        with torch.no_grad():
            image_256 = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
            image_f32 = image_256.float()  # bf16 -> float32 for VAE
            vae_target = vae.encode(2 * image_f32 - 1).latent_dist.mode() * 0.18215  # [B, 4, 32, 32]
            vae_target = vae_target.to(dtype=torch.float16)  # 다시 bf16으로 변환

        loss_l1 = F.l1_loss(outputs["lowlevel_l1"], vae_target)

        # ============ Total Loss (Curriculum: FIC → FIR) ============
        if epoch < fir_start_epoch:
            # Phase 1: FIC + FIM + Lowlevel (방향 학습)
            # loss = (args.fic_weight * loss_fic
            #         + args.fim_weight * loss_fim
            #         + args.lowlevel_weight * (loss_l1/0.18215))
            loss = (args.fir_weight * loss_fir
                    + args.lowlevel_weight * (loss_l1/0.18215))
        else:
            # Phase 2: FIR + FIM + Lowlevel (크기 정밀화, FIC 제외)
            loss = (args.fic_weight * loss_fic
                    + args.fim_weight * loss_fim
                    + args.lowlevel_weight * (loss_l1/0.18215))

        # ============ Backward ============
        # Accelerator가 scaling + gradient sync 자동 처리
        # DeepSpeed: gradient clipping은 deepspeed_config.json에서 설정 (1.0)
        accelerator.backward(loss)

        # Phase 2: DeepSpeed ZeRO-2 gradient buffer workaround
        # DeepSpeed가 prepare() 시점에 등록한 gradient buffer에 NaN이 남으므로 강제 제거
        if epoch >= fir_start_epoch:
            model_unwrapped = accelerator.unwrap_model(model)
            for name, param in model_unwrapped.named_parameters():
                if name.startswith(("region_embedding", "bottleneck")) and param.grad is not None:
                    param.grad = None

        # Phase 2 첫 step: frozen 파라미터 grad 상태 로깅
        if epoch == fir_start_epoch and batch_idx == 0 and accelerator.is_main_process:
            model_unwrapped = accelerator.unwrap_model(model)
            print(f"\n[Freeze Verify] epoch={epoch}:")
            for name, param in model_unwrapped.named_parameters():
                if name.startswith(("region_embedding", "bottleneck")):
                    print(f"  {name}: grad={'None' if param.grad is None else f'norm={param.grad.norm().item():.8f}'}")

        # Optimizer step
        optimizer.step()

        # Learning rate scheduler step
        lr_scheduler.step()

        # ============ Logging ============
        # 모든 프로세스에서 loss 수집
        losses_log.append(loss.item())
        lrs_log.append(optimizer.param_groups[0]['lr'])

        # Main process만 logging
        if accelerator.is_main_process:
            logs = {
                "train/epoch": epoch,
                "train/step": current_step,
                "train/loss": loss.item(),
                "train/loss_fir": loss_fir.item(),
                "train/loss_fic": loss_fic.item(),
                "train/loss_fim": loss_fim.item(),
                "train/loss_l1": loss_l1.item(),
                "train/lr": lrs_log[-1],
            }

            # ============ Debug: NaN/Inf and Value Range Check ============
            # Get unwrapped model for accessing internal parameters
            model_unwrapped = accelerator.unwrap_model(model)

            debug_logs = {
                # Input data
                "debug/fmri_nan": int(torch.isnan(fmri).any().item()),
                "debug/fmri_inf": int(torch.isinf(fmri).any().item()),
                "debug/fmri_min": fmri.min().item(),
                "debug/fmri_max": fmri.max().item(),
                "debug/fmri_mean": fmri.mean().item(),
                # Output embeddings (with mean/std)
                "debug/fmri_proj_nan": int(torch.isnan(outputs["fmri_proj"]).any().item()),
                "debug/fmri_proj_inf": int(torch.isinf(outputs["fmri_proj"]).any().item()),
                "debug/fmri_proj_min": outputs["fmri_proj"].min().item(),
                "debug/fmri_proj_max": outputs["fmri_proj"].max().item(),
                "debug/fmri_proj_mean": outputs["fmri_proj"].mean().item(),
                "debug/fmri_proj_std": outputs["fmri_proj"].std().item(),
                "debug/clip_proj_nan": int(torch.isnan(outputs["clip_proj"]).any().item()),
                "debug/clip_proj_inf": int(torch.isinf(outputs["clip_proj"]).any().item()),
                "debug/clip_proj_min": outputs["clip_proj"].min().item(),
                "debug/clip_proj_max": outputs["clip_proj"].max().item(),
                "debug/clip_proj_mean": outputs["clip_proj"].mean().item(),
                "debug/clip_proj_std": outputs["clip_proj"].std().item(),
                # CLIP vs fMRI comparison
                "debug/clip_fmri_diff_mean": (outputs["clip_proj"] - outputs["fmri_proj"]).abs().mean().item(),
                "debug/clip_fmri_diff_std": (outputs["clip_proj"] - outputs["fmri_proj"]).std().item(),
                "debug/clip_fmri_cosine_sim": F.cosine_similarity(
                    outputs["clip_proj"].reshape(-1, 768),
                    outputs["fmri_proj"].reshape(-1, 768),
                    dim=-1
                ).mean().item(),
                # Low-level decoder output
                "debug/lowlevel_nan": int(torch.isnan(outputs["lowlevel_l1"]).any().item()),
                "debug/lowlevel_inf": int(torch.isinf(outputs["lowlevel_l1"]).any().item()),
                "debug/lowlevel_min": outputs["lowlevel_l1"].min().item(),
                "debug/lowlevel_max": outputs["lowlevel_l1"].max().item(),
                "debug/lowlevel_mean": outputs["lowlevel_l1"].mean().item(),
                # VAE target
                "debug/vae_target_nan": int(torch.isnan(vae_target).any().item()),
                "debug/vae_target_inf": int(torch.isinf(vae_target).any().item()),
                "debug/vae_target_min": vae_target.min().item(),
                "debug/vae_target_max": vae_target.max().item(),
                "debug/vae_target_mean": vae_target.mean().item(),
                # Losses
                "debug/loss_nan": int(torch.isnan(loss).item()),
                "debug/loss_inf": int(torch.isinf(loss).item()),
                "debug/loss_fir_nan": int(torch.isnan(loss_fir).item()),
                "debug/loss_fic_nan": int(torch.isnan(loss_fic).item()),
                "debug/loss_fim_nan": int(torch.isnan(loss_fim).item()),
                "debug/loss_l1_nan": int(torch.isnan(loss_l1).item()),
                "debug/loss_l1_inf": int(torch.isinf(loss_l1).item()),
                # Temperature (FIC loss)
                "debug/fic_temp": model_unwrapped.fic_loss_fn.temp.item(),
                # FIM classifier weight tracking
                "debug/fim_weight_mean": model_unwrapped.fim_classifier.weight.data.mean().item(),
                "debug/fim_weight_std": model_unwrapped.fim_classifier.weight.data.std().item(),
                "debug/fim_weight_norm": model_unwrapped.fim_classifier.weight.data.norm().item(),
                "debug/fim_bias_mean": model_unwrapped.fim_classifier.bias.data.mean().item(),
            }
            logs.update(debug_logs)

            if isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    fir=f"{loss_fir.item():.4f}",
                    fic=f"{loss_fic.item():.4f}",
                    fim=f"{loss_fim.item():.4f}",
                    l1=f"{loss_l1.item():.4f}"
                )

            if args.wandb_log:
                wandb.log(logs, step=current_step)

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    return global_step + len(train_loader), np.mean(losses_log)


# ============================================================================
# Validation Function
# ============================================================================

@torch.no_grad()
def validate(args, model, vae, val_loader, accelerator, epoch, fir_start_epoch):
    """
    Validation loss 계산 (Accelerate DDP 버전)

    모든 프로세스에서 실행 후 샘플 수 기준 가중평균 계산
    Curriculum: epoch < fir_start_epoch이면 FIC, 이후 FIR
    """
    device = accelerator.device
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for fmri, image, subject_name in val_loader:
        batch_size = fmri.size(0)  # 실제 배치 크기 (마지막 배치는 작을 수 있음)
        fmri = fmri.to(device=device, dtype=torch.float16)
        image = image.to(device=device, dtype=torch.float16)

        # Forward (accelerator 전달: FIC/FIM에서 all_gather 사용)
        outputs = model(fmri, image, device, subject_names=subject_name, accelerator=accelerator)

        # Q-Former losses
        loss_fir = outputs["loss_fir"]
        loss_fic = outputs["loss_fic"]
        loss_fim = outputs["loss_fim"]

        # Low-level L1 Loss (VAE는 float32이므로 변환 필요)
        # lowlevel_l1이 [B, 4, 32, 32] (256x256 기준)이므로 image도 256으로 resize
        image_256 = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image_f32 = image_256.float()
        vae_target = vae.encode(2 * image_f32 - 1).latent_dist.mode() * 0.18215  # [B, 4, 32, 32]
        vae_target = vae_target.to(dtype=torch.float16)
        loss_l1 = F.l1_loss(outputs["lowlevel_l1"], vae_target)

        # Total loss (Curriculum: FIC → FIR)
        if epoch < fir_start_epoch:
            # Phase 1: FIC + FIM + Lowlevel (방향 학습)
            # loss = (args.fic_weight * loss_fic
            #         + args.fim_weight * loss_fim
            #         + args.lowlevel_weight * (loss_l1/0.18215))
            loss = (args.fir_weight * loss_fir
                    + args.lowlevel_weight * (loss_l1/0.18215))
        else:
            # Phase 2: FIR + FIM + Lowlevel (크기 정밀화, FIC 제외)
            loss = (args.fic_weight * loss_fic
                    + args.fim_weight * loss_fim
                    + args.lowlevel_weight * (loss_l1/0.18215))

        # 배치 크기로 가중합
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # 모든 GPU에서 gather 후 가중평균
    loss_tensor = torch.tensor([total_loss], device=device)
    samples_tensor = torch.tensor([total_samples], device=device, dtype=torch.long)

    all_losses = accelerator.gather(loss_tensor)      # [sum0, sum1, sum2, sum3]
    all_samples = accelerator.gather(samples_tensor)  # [n0, n1, n2, n3]

    avg_loss = all_losses.sum().item() / all_samples.sum().item()

    model.train()
    return avg_loss


# ============================================================================
# Evaluation Function (Reconstruction) - DDP Distributed
# ============================================================================

@torch.no_grad()
def evaluate(args, model, vae, versatile_diffusion, test_loader, epoch, subject, recon_dir, low_recon_dir, accelerator):
    """
    Evaluation: fMRI -> 이미지 reconstruction (DDP로 병렬 처리)

    Args:
        recon_dir: Versatile Diffusion 결과 저장 디렉토리 (실험이름_epoch*/recon/)
        low_recon_dir: Low-level (VAE blurry) 결과 저장 디렉토리 (실험이름_epoch*/low_recon/)

    주의: 모든 프로세스에서 호출되며, 각 GPU가 다른 batch를 처리
    """
    device = accelerator.device
    rank = accelerator.process_index

    # DDP unwrap
    model_unwrapped = accelerator.unwrap_model(model)
    model_unwrapped.eval()

    # Generator for reproducibility (각 GPU마다 동일한 seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    num_samples = 0

    # Progress bar는 main process만 표시
    if accelerator.is_main_process:
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader),
                            desc=f"Evaluation Epoch {epoch} [{subject}]", ncols=120)
    else:
        progress_bar = enumerate(test_loader)

    for batch_idx, (fmri, gt_images, image_ids, subject_name) in progress_bar:
        # Data -> GPU with correct dtype
        fmri = fmri.to(device=device, dtype=torch.float16)

        # ============ Forward Inference ============
        outputs = model_unwrapped.inference(fmri, subject_names=subject_name)
        fmri_proj = outputs["fmri_proj"]      # [B, 257, 768]
        lowlevel_l1 = outputs["lowlevel_l1"]  # [B, 4, 32, 32] (256x256 기준)

        # ============ VAE Decode (Blurry Image for saving) ============
        # lowlevel_l1: [B, 4, 32, 32] → VAE Decode → [B, 3, 256, 256]
        lowlevel_f32 = (lowlevel_l1 / 0.18215).float()
        blurry_image = vae.decode(lowlevel_f32).sample / 2 + 0.5  # [B, 3, 256, 256]
        blurry_image = blurry_image.clamp(0, 1)

        # ============ Versatile Diffusion용 latent 생성 ============
        # 256x256 → 512x512 resize → VAE Encode → [B, 4, 64, 64]
        blurry_512 = F.interpolate(blurry_image, (512, 512), mode='bilinear', align_corners=False)
        blurry_512_f32 = blurry_512.float()
        init_latents_64 = vae.encode(2 * blurry_512_f32 - 1).latent_dist.mode() * 0.18215  # [B, 4, 64, 64]

        # ============ Versatile Diffusion Reconstruction ============
        recon_tensors = versatile_diffusion_reconstruct(
            versatile_diffusion,
            init_latents_64,  # [B, 4, 64, 64] (512x512 기준)
            fmri_proj,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            img2img_strength=args.img2img_strength,
            generator=generator,
        )  # [B, 3, 512, 512]

        # ============ Ground Truth 이미지 ============
        gt_tensors = gt_images

        # ============ 배치 단위로 디스크에 저장 ============
        # 1. Versatile Diffusion 결과 (recon/)
        save_recon_batch(recon_tensors, gt_tensors, image_ids, recon_dir, subject)

        # 2. Low-level (VAE blurry) 결과 (low_recon/)
        save_lowlevel_batch(blurry_image, gt_tensors, image_ids, low_recon_dir, subject)

        num_samples += len(image_ids)

        # 메모리 정리
        del recon_tensors, gt_tensors, fmri_proj, lowlevel_l1, blurry_image

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    return num_samples


# ============================================================================
# Metric Evaluation - Main Process Only
# ============================================================================

def compute_metrics(args, all_recons, all_targets, metrics):
    """
    Reconstruction 품질 평가 (Main Process Only)

    Metrics:
        - PixCorr, SSIM (low-level)
        - AlexNet_2, AlexNet_5 (mid-level)
        - CLIP, Inception, EfficientNet, SwAV (high-level)
    """
    results = {}

    print("\nComputing metrics...")

    # Low-level metrics
    results["PixCorr"] = metrics["pixcorr"](all_recons, all_targets)
    results["SSIM"] = metrics["ssim"](all_recons, all_targets)

    # AlexNet features
    results["AlexNet_2"] = metrics["alexnet2"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["alexnet2"]["model"],
        metrics["alexnet2"]["preprocess"],
        metrics["alexnet2"]["layer"]
    )

    results["AlexNet_5"] = metrics["alexnet5"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["alexnet5"]["model"],
        metrics["alexnet5"]["preprocess"],
        metrics["alexnet5"]["layer"]
    )

    # High-level semantic metrics
    results["CLIP"] = metrics["clip"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["clip"]["model"],
        metrics["clip"]["preprocess"]
    )

    results["Inception"] = metrics["inception"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["inception"]["model"],
        metrics["inception"]["preprocess"]
    )

    results["EfficientNet"] = metrics["efficientnet"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["efficientnet"]["model"],
        metrics["efficientnet"]["preprocess"]
    )

    results["SwAV"] = metrics["swav"]["metric_fn"](
        args, all_recons, all_targets,
        metrics["swav"]["model"],
        metrics["swav"]["preprocess"]
    )

    return results


