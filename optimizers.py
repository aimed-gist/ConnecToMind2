import torch


def adamw(parameters, lr):
    """AdamW optimizer"""
    return torch.optim.AdamW(parameters, lr=lr)


# q former pretrained된 부분 lr 다르게 주기
def get_optimizer_with_different_lr(args, model, backbone_lr_scale=0.1):
    """
    서로 다른 learning rate를 사용하는 optimizer

    Q-Former의 CLIP-initialized 부분은 낮은 lr 사용

    Args:
        args: arguments
        model: ConnecToMind2 model
        backbone_lr_scale: Q-Former backbone의 lr 스케일 (default: 0.1)

    Returns:
        optimizer: configured optimizer
    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'ln']

    opt_grouped_parameters = [
        # Region Embedding (full lr - randomly initialized)
        {
            'params': [p for n, p in model.region_embedding.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.max_lr
        },
        {
            'params': [p for n, p in model.region_embedding.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.max_lr
        },
        # Connectome-QFormer (lower lr - CLIP initialized)
        {
            'params': [p for n, p in model.connectome_qformer.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.max_lr * backbone_lr_scale
        },
        {
            'params': [p for n, p in model.connectome_qformer.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.max_lr * backbone_lr_scale
        },
        # Output Projection (full lr - randomly initialized)
        {
            'params': [p for n, p in model.output_proj.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.max_lr
        },
        {
            'params': [p for n, p in model.output_proj.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.max_lr
        },
        # Low-Level Decoder (full lr - randomly initialized)
        {
            'params': [p for n, p in model.low_level_decoder.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.max_lr
        },
        {
            'params': [p for n, p in model.low_level_decoder.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.max_lr
        },
        # FIM Classifier (full lr - randomly initialized)
        {
            'params': [p for n, p in model.fim_classifier.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
            'lr': args.max_lr
        },
        {
            'params': [p for n, p in model.fim_classifier.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': args.max_lr
        },
        # FIC Loss Temperature (full lr - learnable parameter)
        {
            'params': [model.fic_loss_fn.temp],
            'weight_decay': 0.0,
            'lr': args.max_lr
        },
    ]

    return adamw(opt_grouped_parameters, lr=args.max_lr)
