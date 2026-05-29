"""
ConnecToMind2 Main Entry Point with Accelerate (DeepSpeed ZeRO-2)

실행 방법:
    # Single GPU (테스트용)
    python main_accelerate.py

    # Multi-GPU (DeepSpeed ZeRO-2) - config 파일 사용 (권장)
    accelerate launch --config_file configs/accelerate_config.yaml main_accelerate.py

    # Multi-GPU (DeepSpeed ZeRO-2) - 커맨드라인
    accelerate launch --mixed_precision=bf16 --use_deepspeed --num_processes=6 main_accelerate.py

주의사항:
    - batch_size는 GPU당 배치 크기입니다
    - Total effective batch = batch_size × num_processes
    - Learning rate는 자동으로 scaling되지 않으므로 수동 조정 필요
    - DeepSpeed ZeRO-2: optimizer state + gradient를 GPU 간 분산 (메모리 절감 50-60%)
"""

import os

# NCCL 타임아웃 설정 (기본 30분 -> 2시간)
# 반드시 torch import 전에 설정해야 적용됨!
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import math
import numpy as np
import torch
import wandb

from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed

from args import parse_args
from data import get_dataloader
from models import get_model  # Bottleneck MLP 추가 버전
# from models import get_model  # 원본 모델
# from models_7net import get_model
from optimizers import get_optimizer_with_different_lr
from schedulers import get_scheduler
from metrics import get_metric
from trainer_accelerate import train_evaluate_metric


def main():
    """
    실행 순서:
        1. Accelerator 초기화 (DDP 설정)
        2. args 파싱
        3. Seed 설정 (재현성)
        4. WandB 초기화 (main process only)
        5. 데이터 로더 생성
        6. 모델 로드
        7. Optimizer, Scheduler 생성
        8. Accelerator prepare (DDP wrapping)
        9. Metric 모델 로드 (main process only)
        10. 학습 시작
    """

    # ============ 1. Accelerator 초기화 ============
    # DeepSpeed ZeRO-2: accelerate_config.yaml에서 설정 로드
    # - gradient_clipping: deepspeed_config.json에서 설정 (1.0)
    # - mixed_precision: accelerate_config.yaml에서 설정 (bf16)
    # - zero_stage: 2 (optimizer state + gradient 분산)

    # NCCL 타임아웃: 3시간 (metric 계산 시간 고려)
    # 이 방법이 환경변수보다 확실함
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb" if False else None,  # WandB tracking (optional)
        kwargs_handlers=[process_group_kwargs],
    )

    # ============ 2. Args ============
    args = parse_args()

    # Accelerator에서 device 자동 할당 (각 process마다 다른 GPU)
    args.device = accelerator.device

    # Learning rate scaling 비활성화 (기본값 유지)
    # args.max_lr = args.max_lr * math.sqrt(accelerator.num_processes)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("ConnecToMind2 Configuration (Accelerate DDP)")
        print(f"{'='*60}")
        print(f"  Mode: {args.mode}")
        print(f"  Distributed: {accelerator.num_processes} processes")
        print(f"  Local Rank: {accelerator.local_process_index}")
        print(f"  Device: {args.device}")
        print(f"  Mixed Precision: {accelerator.mixed_precision}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch Size (per GPU): {args.batch_size}")
        print(f"  Total Batch Size: {args.batch_size * accelerator.num_processes}")
        print(f"  Learning Rate: {args.max_lr}")
        print(f"  Subjects: {args.subjects}")
        print(f"  Experiment: {args.experiment_name}")
        print(f"{'='*60}\n")

    # ============ 3. Seed 설정 ============
    set_seed(args.seed)
    if accelerator.is_main_process:
        print(f"Random seed set to {args.seed}")

    # ============ 4. WandB 초기화 (main process only) ============
    if args.wandb_log and accelerator.is_main_process:
        wandb.init(
            project="ConnecToMind2",
            name=args.experiment_name,
            config=vars(args)
        )
        print("WandB initialized")

    # ============ 5. 데이터 로더 생성 ============
    if accelerator.is_main_process:
        print("\nLoading data...")

    # Train + Validation loader
    args.mode = 'train'
    train_loader, val_loader = get_dataloader(args)

    if accelerator.is_main_process:
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")

    # Test loaders (subject별 분리) - DDP로 평가 병렬화
    args.mode = 'inference'
    # Evaluation용 num_workers 줄임 (NCCL 타임아웃 방지)
    original_num_workers = args.num_workers
    args.num_workers = args.eval_num_workers
    test_loaders_raw = get_dataloader(args)
    args.num_workers = original_num_workers  # 복원

    # Test loaders를 DDP로 준비 (각 GPU가 다른 batch 처리)
    test_loaders = {}
    for sub, loader in test_loaders_raw.items():
        test_loaders[sub] = accelerator.prepare(loader)

    if accelerator.is_main_process:
        total_test = sum(len(loader.dataset) for loader in test_loaders_raw.values())
        print(f"  Test samples: {total_test} (subject별: {', '.join(f'{sub}={len(loader.dataset)}' for sub, loader in test_loaders_raw.items())})")
        print(f"  Evaluation will use DDP with {accelerator.num_processes} GPUs (num_workers={args.eval_num_workers})")

    # ============ 6. 모델 로드 ============
    if accelerator.is_main_process:
        print("\nLoading models...")

    models = get_model(args)

    if accelerator.is_main_process:
        print("  ConnecToMind2 model loaded")
        print("  Versatile Diffusion pipeline loaded")
        print("  VAE loaded")

    # ============ 7. Optimizer & Scheduler ============
    if accelerator.is_main_process:
        print("\nSetting up optimizer and scheduler...")

    optimizer = get_optimizer_with_different_lr(args, models["connectomind2"])
    lr_scheduler = get_scheduler(
        args, optimizer, len(train_loader.dataset),
        num_processes=accelerator.num_processes  # DDP GPU 수 전달
    )

    if accelerator.is_main_process:
        print(f"  Optimizer: {args.optimizer}")
        print(f"  Scheduler: {args.scheduler_type}")

    # ============ 8. Accelerator Prepare ============
    # DDP로 모델, optimizer, dataloader wrap
    # - model: DistributedDataParallel로 wrap
    # - dataloader: DistributedSampler 자동 적용
    # - optimizer: gradient synchronization 설정
    models["connectomind2"], optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        models["connectomind2"], optimizer, train_loader, val_loader, lr_scheduler
    )

    # VAE, Versatile Diffusion은 frozen이므로 각 GPU에 복사만
    # (학습되지 않으므로 DDP 불필요)
    models["vae"] = models["vae"].to(accelerator.device)
    if models["versatile_diffusion"] is not None:
        models["versatile_diffusion"] = models["versatile_diffusion"].to(accelerator.device)

    if accelerator.is_main_process:
        print("  Models prepared for distributed training")
        print(f"  Trainable parameters: {sum(p.numel() for p in models['connectomind2'].parameters() if p.requires_grad):,}")

    # ============ 9. Resume from checkpoint ============
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("RESUME FROM CHECKPOINT")
            print(f"{'='*60}")
            print(f"  Checkpoint file: {args.resume_checkpoint}")

        # 체크포인트 디렉토리 확인
        checkpoint_dir = os.path.dirname(args.resume_checkpoint)
        accelerate_ckpt_dir = os.path.join(checkpoint_dir, "accelerate_checkpoint")

        # ============ 디버깅: 파일 존재 확인 ============
        if accelerator.is_main_process:
            print(f"\n[DEBUG] Checking files...")
            print(f"  Checkpoint dir: {checkpoint_dir}")
            print(f"  Accelerate dir: {accelerate_ckpt_dir}")
            print(f"  Checkpoint exists: {os.path.exists(args.resume_checkpoint)}")
            print(f"  Accelerate dir exists: {os.path.exists(accelerate_ckpt_dir)}")
            if os.path.exists(accelerate_ckpt_dir):
                files = os.listdir(accelerate_ckpt_dir)
                print(f"  Accelerate files: {files}")

        # 메타데이터 로드 (epoch, global_step 등)
        checkpoint = torch.load(args.resume_checkpoint, map_location=accelerator.device)

        if accelerator.is_main_process:
            print(f"\n[DEBUG] Checkpoint metadata:")
            print(f"  Saved epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Saved global_step: {checkpoint.get('global_step', 'N/A')}")
            print(f"  Saved train_loss: {checkpoint.get('train_loss', 'N/A')}")
            print(f"  Saved val_loss: {checkpoint.get('val_loss', 'N/A')}")
            print(f"  Saved frozen_pre_qformer: {checkpoint.get('frozen_pre_qformer', 'N/A')}")

        # ============ 디버깅: Load 전 상태 ============
        if accelerator.is_main_process:
            print(f"\n[DEBUG] Before loading accelerator state:")
            print(f"  LR (before): {optimizer.param_groups[0]['lr']:.2e}")
            unwrapped = accelerator.unwrap_model(models["connectomind2"])
            sample_param = list(unwrapped.parameters())[0]
            print(f"  Sample param mean (before): {sample_param.data.mean().item():.6f}")
            print(f"  Sample param std (before): {sample_param.data.std().item():.6f}")

        # DeepSpeed 호환 형식으로 로드 (accelerator.save_state()로 저장된 것)
        accelerator.wait_for_everyone()
        accelerator.load_state(accelerate_ckpt_dir)
        accelerator.wait_for_everyone()

        # ============ 디버깅: Load 후 상태 ============
        if accelerator.is_main_process:
            print(f"\n[DEBUG] After loading accelerator state:")
            print(f"  LR (after): {optimizer.param_groups[0]['lr']:.2e}")
            unwrapped = accelerator.unwrap_model(models["connectomind2"])
            sample_param = list(unwrapped.parameters())[0]
            print(f"  Sample param mean (after): {sample_param.data.mean().item():.6f}")
            print(f"  Sample param std (after): {sample_param.data.std().item():.6f}")

        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * len(train_loader))

        # ============ 디버깅: 최종 resume 정보 ============
        if accelerator.is_main_process:
            print(f"\n[DEBUG] Resume summary:")
            print(f"  Resumed from epoch: {checkpoint['epoch']}")
            print(f"  Starting epoch: {start_epoch}")
            print(f"  Global step: {global_step}")
            print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Scheduler 상태 확인
            if hasattr(lr_scheduler, 'last_epoch'):
                print(f"  Scheduler last_epoch: {lr_scheduler.last_epoch}")
            if hasattr(lr_scheduler, '_step_count'):
                print(f"  Scheduler step_count: {lr_scheduler._step_count}")

            print(f"{'='*60}\n")

        # frozen_pre_qformer 상태 복원
        frozen_pre_qformer = checkpoint.get('frozen_pre_qformer', False)
    else:
        frozen_pre_qformer = False

    # ============ 10. Metric 모델 로드 (main process only) ============
    metrics = None
    if accelerator.is_main_process:
        print("\nLoading metric models...")
        metrics = get_metric(args)
        print("  Metrics loaded: PixCorr, SSIM, AlexNet, CLIP, Inception, EfficientNet, SwAV")

    # ============ 11. 학습 시작 ============
    if accelerator.is_main_process:
        print("\nStarting training...")

    trained_model = train_evaluate_metric(
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loaders=test_loaders,  # subject별 분리된 dict
        models=models,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metrics=metrics,
        accelerator=accelerator,
        start_epoch=start_epoch,  # resume용
        start_global_step=global_step,  # resume용
        resume_frozen_pre_qformer=frozen_pre_qformer  # resume용
    )

    # ============ 12. WandB 종료 ============
    if args.wandb_log and accelerator.is_main_process:
        wandb.finish()
        print("WandB finished")

    if accelerator.is_main_process:
        print("\nTraining Complete!")


if __name__ == "__main__":
    main()
