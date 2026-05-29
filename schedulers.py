import torch
import math


def linear_scheduler(optimizer, total_steps):
    """LinearLR: 학습률을 선형적으로 감소"""
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=total_steps,
        last_epoch=-1
    )


def cycle_scheduler(optimizer, total_steps, max_lr, num_epochs):
    """OneCycleLR: warmup + decay 사이클 학습률"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        div_factor=3,           # 시작 LR = max_lr/3 = 0.0001
        # final_div_factor=2,   # 최종 LR = 0.00005
        # final_div_factor=10,  # 최종 LR = 0.00001
        final_div_factor=1000,  # 최종 LR = 0.0001/1000 = 0.0000001
        # div_factor=25,      # (기존 기본값)
        # final_div_factor=1000,  # (기존) 최저 LR = max_lr/25/1000 = 0.000000012
        # final_div_factor=6,  # (기존) 최저 LR = max_lr/25/6 = 0.000002
        last_epoch=-1,
        pct_start=2 / num_epochs  # warmup 비율
    )


def cosine_scheduler(optimizer, total_steps, num_warmup_steps=0):
    """CosineAnnealingLR with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cosine_constant_scheduler(optimizer, total_steps, max_lr, num_epochs, decay_epochs=40):
    """
    OneCycleLR + ConstantLR
    - epoch 0~decay_epochs: OneCycle (warmup → max_lr → min_lr)
    - epoch decay_epochs~num_epochs: constant (min_lr 유지)

    OneCycleLR 설정:
        - 시작 LR: 0.0001 (max_lr / div_factor)
        - 최고 LR: 0.0003 (max_lr)
        - 최저 LR: 1.2e-8 (start_lr / final_div_factor)

    Args:
        decay_epochs: OneCycle이 끝나는 epoch (기본값: 40)
    """
    min_lr = 1.2e-8  # 최저 LR

    # decay_epochs까지의 step 수 계산
    steps_per_epoch = total_steps // num_epochs
    decay_steps = decay_epochs * steps_per_epoch

    # Phase 1: OneCycleLR (0 ~ decay_epochs)
    # start_lr = max_lr / div_factor = 0.0003 / 3 = 0.0001
    # end_lr = start_lr / final_div_factor = 0.0001 / 8333.33 ≈ 1.2e-8
    onecycle = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=decay_steps,
        div_factor=3,              # 시작 LR = 0.0003/3 = 0.0001
        final_div_factor=8333.33,  # 최저 LR = 0.0001/8333.33 ≈ 1.2e-8
        pct_start=2 / decay_epochs,  # warmup 비율 (2 epoch)
        last_epoch=-1
    )

    # Phase 2: LambdaLR로 min_lr 고정
    # NOTE: OneCycleLR 생성 후 optimizer의 initial_lr이 변경되므로,
    # LambdaLR에서 initial_lr로 나눠서 상쇄시켜야 정확한 min_lr이 나옴
    # lr = initial_lr * (min_lr / initial_lr) = min_lr
    initial_lr = optimizer.param_groups[0]['initial_lr']
    constant = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: min_lr / initial_lr
    )

    # SequentialLR로 연결
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[onecycle, constant],
        milestones=[decay_steps]
    )


def get_scheduler(args, optimizer, num_train, num_processes=1):
    """
    학습률 스케줄러 선택자

    Args:
        args: arguments containing scheduler settings
        optimizer: optimizer instance
        num_train: number of training samples
        num_processes: DDP에서 사용하는 GPU 수 (default: 1)

    Returns:
        scheduler: learning rate scheduler
    """
    # total_steps 계산
    # NOTE: DeepSpeed에서 각 GPU가 독립적으로 scheduler.step()을 호출하므로
    # total_steps를 num_processes배로 설정해야 올바른 epoch 기준 동작
    # (실제 global step당 num_processes번 step됨)
    effective_batch_size = args.batch_size * num_processes
    steps_per_epoch = math.ceil(num_train / effective_batch_size)
    total_steps = int(args.num_epochs * steps_per_epoch * num_processes)

    if args.scheduler_type == 'linear':
        return linear_scheduler(optimizer, total_steps)

    if args.scheduler_type == 'cycle':
        return cycle_scheduler(optimizer, total_steps, args.max_lr, args.num_epochs)

    if args.scheduler_type == 'cosine':
        num_warmup_steps = int(total_steps * 0.1)  # 10% warmup
        return cosine_scheduler(optimizer, total_steps, num_warmup_steps)

    if args.scheduler_type == 'cosine_constant':
        return cosine_constant_scheduler(
            optimizer, total_steps, args.max_lr, args.num_epochs,
            decay_epochs=args.decay_epochs
        )

    raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")
