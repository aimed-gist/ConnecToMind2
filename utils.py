"""
ConnecToMind2 Utility Functions
"""

import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt


def img_augment(image, num_aug=1):
    """
    이미지 augmentation

    Args:
        image: [B, 3, H, W] tensor
        num_aug: augmentation 횟수

    Returns:
        augmented image tensor
    """
    augmented = []
    for _ in range(num_aug):
        img = image
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[-1])
        # Color jitter (brightness, contrast)
        img = img * (0.8 + 0.4 * torch.rand(1, device=image.device))
        img = img.clamp(0, 1)
        augmented.append(img)
    return torch.cat(augmented, dim=0) if num_aug > 1 else augmented[0]


def get_unique_path(path):
    """
    중복 파일명 방지

    Args:
        path: 원본 파일 경로

    Returns:
        unique path (중복 시 _1, _2, ... 추가)
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def save_reconstructions(save_dict, save_dir):
    """
    Reconstruction 이미지 저장 (Legacy - 호환성 유지)

    Args:
        save_dict: {image_id: (recon_tensor, gt_tensor)} dictionary
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    for img_id, (recon_img, gt_img) in save_dict.items():
        # Reconstruction
        recon_path = os.path.join(save_dir, f"recon_{img_id}")
        save_image(recon_img.clamp(0, 1), recon_path)
        # Ground Truth
        gt_path = os.path.join(save_dir, f"gt_{img_id}")
        save_image(gt_img.clamp(0, 1), gt_path)


def save_recon_batch(recon_tensors, gt_tensors, image_ids, save_dir, subject):
    """
    GT vs Recon 비교 이미지 저장 (MindEye1 스타일) + 개별 이미지 (metric용)

    Args:
        recon_tensors: [B, 3, H, W] reconstruction 이미지 텐서
        gt_tensors: [B, 3, H, W] GT 이미지 텐서
        image_ids: List of image IDs
        save_dir: 기본 저장 디렉토리
        subject: Subject ID (예: 'sub-01')
    """
    subject_dir = os.path.join(save_dir, subject)
    compare_dir = os.path.join(save_dir, f"{subject}_compare")
    os.makedirs(subject_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    for i, img_id in enumerate(image_ids):
        base_name = os.path.splitext(img_id)[0]

        # 개별 이미지 저장 (metric 계산용)
        save_image(recon_tensors[i].clamp(0, 1), os.path.join(subject_dir, f"{base_name}_recon.png"))
        save_image(gt_tensors[i].clamp(0, 1), os.path.join(subject_dir, f"{base_name}_gt.png"))

        # GT vs Recon 나란히 저장 (시각화용)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_tensors[i].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        axes[1].imshow(recon_tensors[i].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(compare_dir, f"{base_name}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


def save_lowlevel_batch(lowlevel_tensors, gt_tensors, image_ids, save_dir, subject):
    """
    GT vs Low-level 비교 이미지 저장 (MindEye1 스타일) + 개별 이미지 (metric용)

    Args:
        lowlevel_tensors: [B, 3, H, W] low-level 이미지 텐서 (VAE decoded)
        gt_tensors: [B, 3, H, W] GT 이미지 텐서
        image_ids: List of image IDs
        save_dir: 기본 저장 디렉토리
        subject: Subject ID (예: 'sub-01')
    """
    subject_dir = os.path.join(save_dir, subject)
    compare_dir = os.path.join(save_dir, f"{subject}_compare")
    os.makedirs(subject_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    for i, img_id in enumerate(image_ids):
        base_name = os.path.splitext(img_id)[0]

        # 개별 이미지 저장 (metric 계산용)
        save_image(lowlevel_tensors[i].clamp(0, 1), os.path.join(subject_dir, f"{base_name}_lowlevel.png"))
        save_image(gt_tensors[i].clamp(0, 1), os.path.join(subject_dir, f"{base_name}_gt.png"))

        # GT vs Lowlevel 나란히 저장 (시각화용)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_tensors[i].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        axes[1].imshow(lowlevel_tensors[i].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[1].set_title("Low-level (Blurry)")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(compare_dir, f"{base_name}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


def load_recons_from_disk(save_dir, subject):
    """
    디스크에서 Subject별 reconstruction/GT 이미지를 로드

    Args:
        save_dir: 기본 저장 디렉토리
        subject: Subject ID (예: 'sub-01')

    Returns:
        recons: [N, 3, H, W] reconstruction 이미지 텐서
        targets: [N, 3, H, W] GT 이미지 텐서
    """
    from torchvision import transforms
    from natsort import natsorted

    subject_dir = os.path.join(save_dir, subject)
    to_tensor = transforms.ToTensor()

    # recon 파일 목록 (정렬)
    recon_files = natsorted([f for f in os.listdir(subject_dir) if f.endswith('_recon.png')])

    recons = []
    targets = []

    for recon_file in recon_files:
        # recon 로드
        recon_path = os.path.join(subject_dir, recon_file)
        recon_img = Image.open(recon_path).convert('RGB')
        recons.append(to_tensor(recon_img))

        # gt 로드 (recon 파일명에서 gt 파일명 생성)
        gt_file = recon_file.replace('_recon.png', '_gt.png')
        gt_path = os.path.join(subject_dir, gt_file)
        gt_img = Image.open(gt_path).convert('RGB')
        targets.append(to_tensor(gt_img))

    recons = torch.stack(recons, dim=0)
    targets = torch.stack(targets, dim=0)

    return recons, targets


def load_gt_image(image_dir, image_id, transform=None):
    """
    Ground Truth 이미지 로드

    Args:
        image_dir: 이미지 디렉토리 경로
        image_id: 이미지 파일명 (예: 'coco2017_14.jpg')
        transform: 이미지 transform (default: ToTensor)

    Returns:
        image tensor [3, H, W]
    """
    from torchvision import transforms

    if transform is None:
        transform = transforms.ToTensor()

    image_path = os.path.join(image_dir, image_id)
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def load_gt_images_batch(image_dir, image_ids, transform=None):
    """
    여러 Ground Truth 이미지를 배치로 로드

    Args:
        image_dir: 이미지 디렉토리 경로
        image_ids: 이미지 파일명 리스트
        transform: 이미지 transform

    Returns:
        image tensor [B, 3, H, W]
    """
    images = [load_gt_image(image_dir, img_id, transform) for img_id in image_ids]
    return torch.stack(images, dim=0)


# ============================================================================
# Versatile Diffusion Reconstruction (MindEye1 방식)
# ============================================================================

@torch.no_grad()
def versatile_diffusion_reconstruct(
    vd_pipe,
    init_latents,
    brain_clip_embeddings,
    num_inference_steps=20,
    guidance_scale=7.5,
    img2img_strength=0.85,
    generator=None,
):
    """
    Versatile Diffusion을 사용한 이미지 재구성 (MindEye1 utils.reconstruction 참고)

    Args:
        vd_pipe: Versatile Diffusion pipeline (DiffusionPipeline)
        init_latents: [B, 4, 64, 64] VAE latent (LowLevelDecoder에서 직접 생성)
        brain_clip_embeddings: [B, 257, 768] fMRI에서 예측한 CLIP embedding
        num_inference_steps: diffusion steps
        guidance_scale: classifier-free guidance scale
        img2img_strength: img2img 강도 (1=no img2img, 0=only lowlevel)
        generator: torch.Generator for reproducibility

    Returns:
        List of PIL images
    """
    device = brain_clip_embeddings.device
    batch_size = brain_clip_embeddings.shape[0]

    # Versatile Diffusion 컴포넌트 추출
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    # Noise scheduler 설정
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    # CLS Token Normalization (MindEye1 스타일)
    brain_clip_embeddings = brain_clip_embeddings.clone()
    for samp in range(batch_size):
        cls_norm = brain_clip_embeddings[samp, 0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6
        brain_clip_embeddings[samp] = brain_clip_embeddings[samp] / cls_norm

    # Classifier-free guidance를 위한 embedding 준비
    # uncond: zeros, cond: brain_clip_embeddings
    do_classifier_free_guidance = guidance_scale > 1.0

    if do_classifier_free_guidance:
        uncond_embeddings = torch.zeros_like(brain_clip_embeddings)
        input_embedding = torch.cat([uncond_embeddings, brain_clip_embeddings], dim=0)  # [2B, 257, 768]
    else:
        input_embedding = brain_clip_embeddings

    # img2img: init_latents를 초기 latent로 직접 사용 (VAE encode 불필요)
    if init_latents is not None and img2img_strength < 1.0:
        # init_latents [B, 4, 64, 64] - LowLevelDecoder에서 직접 생성
        # Training 시 0.18215 scale로 학습되었으므로 그대로 사용
        # FP32: VD를 fp32로 사용

        # img2img timestep 계산
        init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1].repeat(batch_size)

        # 노이즈 추가
        noise = torch.randn(init_latents.shape, device=device, generator=generator, dtype=input_embedding.dtype)
        latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
    else:
        # 순수 noise에서 시작
        timesteps = noise_scheduler.timesteps
        latents = torch.randn([batch_size, 4, 64, 64], device=device, generator=generator, dtype=input_embedding.dtype)
        latents = latents * noise_scheduler.init_noise_sigma

    # Denoising loop
    for t in timesteps:
        # Classifier-free guidance를 위해 latent 복제
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        # UNet forward (FP32) - dtype 맞춤
        noise_pred = unet(latent_model_input.float(), t, encoder_hidden_states=input_embedding.float()).sample

        # Classifier-free guidance 적용
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step (FP32) - latents도 float32로 변환
        latents = noise_scheduler.step(noise_pred, t, latents.float()).prev_sample

    # VAE decode: latent -> image (FP32)
    latents = latents / 0.18215
    images_list = []
    chunck_size = 2
    for i in range(0, batch_size, chunck_size):
        chunk_latents = latents[i:i+chunck_size]
        with torch.no_grad():
            decoded_images = vae.decode(chunk_latents.float()).sample
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
            images_list.append(decoded_images.cpu())
    images = torch.cat(images_list, dim=0)

    # Tensor 그대로 반환 (PIL 변환 제거)
    return images  # [B, 3, 512, 512]
