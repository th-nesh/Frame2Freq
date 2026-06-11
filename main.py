import os
import argparse
import math
import json

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from contextlib import nullcontext
from utils import setup_for_distributed, MetricLogger, SmoothedValue, load_model, save_model
import models_adapter
from video_dataset import VideoDataset
from configs import DATASETS

def init_dist():
    """
    Initialize torch.distributed for either SLURM+srun or torchrun.
    - If torchrun: uses env:// (expects RANK/WORLD_SIZE/LOCAL_RANK).
    - If srun:     uses TCP store with MASTER_ADDR/MASTER_PORT and SLURM_* vars.
    Returns local_rank (GPU index on this node).
    """
    if dist.is_initialized():
        # Already initialized (e.g., by wrapper); just set device and return.
        local_rank = int(os.environ.get("LOCAL_RANK",
                          os.environ.get("SLURM_LOCALID", 0)))
        torch.cuda.set_device(local_rank)
        return local_rank

    # Detect launcher
    have_torchrun = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    have_slurm    = all(k in os.environ for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))

    if have_torchrun:
        # torchrun path
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank(); world = dist.get_world_size()
    elif have_slurm:
        # srun path (what you’re using)
        rank       = int(os.environ["SLURM_PROCID"])
        world      = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if not master_addr or not master_port:
            raise RuntimeError("MASTER_ADDR/MASTER_PORT must be exported when using srun.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world,
        )
    else:
        raise RuntimeError(
            "Could not detect launcher. Use torchrun (RANK/WORLD_SIZE/LOCAL_RANK) "
            "or SLURM srun (SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID + MASTER_*)."
        )

    # Optional debug
    node = os.uname()[1]
    print(f"[DDP] host={node} rank={rank} local_rank={local_rank} world={world}", flush=True)

    dist.barrier()
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)   # per-GPU
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--auto_remove', action='store_true')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrain', type=str)

    parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys())
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--spatial_size', type=int, default=224)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--sampling_rate', type=int, default=0)
    parser.add_argument('--num_spatial_views', type=int, default=1, choices=[1, 3])
    parser.add_argument('--num_temporal_views', type=int, default=1)
    parser.add_argument('--auto_augment', type=str)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resize_type', type=str, default='random_resized_crop',
                        choices=['random_resized_crop', 'random_short_side_scale_jitter'])
    parser.add_argument('--scale_range', type=float, nargs=2, default=[0.08, 1.0])
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--visualize_attn', action='store_true',
                        help='Generate attention overlays on the first validation clip when running evaluation-only.')
    parser.add_argument('--visualize_attn_batches', type=int, default=50,
                        help='Number of validation batches to capture attention overlays for (1=first batch only).')

    args = parser.parse_args()

    # --- DDP init + GPU binding ---
    local_rank = init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    setup_for_distributed(rank == 0)

    if rank == 0:
        print("{}".format(args).replace(', ', ',\n'), flush=True)

    torch.backends.cudnn.benchmark = True  # perf

    # --- Model ---
    if rank == 0:
        print('creating model', flush=True)
    model = models_adapter.__dict__[args.model](num_classes=DATASETS[args.dataset]['NUM_CLASSES']).cuda()
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'Trainable param: {n}, {tuple(p.size())}, {p.dtype}', flush=True)
        print('Total trainable params:', n_trainable_params, f'({n_trainable_params/1e6:.2f} M)', flush=True)

    # Bind DDP to the current GPU
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )
    model_without_ddp = model.module

    # --- Datasets / Loaders ---
    if rank == 0:
        print('creating dataset', flush=True)

    if not args.eval_only:
        dataset_train = VideoDataset(
            list_path=DATASETS[args.dataset]['TRAIN_LIST'],
            data_root=DATASETS[args.dataset]['TRAIN_ROOT'],
            random_sample=True,
            mirror=args.mirror,
            spatial_size=args.spatial_size,
            auto_augment=args.auto_augment,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            resize_type=args.resize_type,
            scale_range=args.scale_range,
        )
        if rank == 0:
            print('train dataset:', dataset_train, flush=True)

    dataset_val = VideoDataset(
        list_path=DATASETS[args.dataset]['VAL_LIST'],
        data_root=DATASETS[args.dataset]['VAL_ROOT'],
        random_sample=False,
        spatial_size=args.spatial_size,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
    )
    if rank == 0:
        print('val dataset:', dataset_val, flush=True)

    if not args.eval_only:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    # shard val set by rank
    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset_val, range(rank, len(dataset_val), world)),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # --- Optimizer / LR ---
    if args.eval_only:
        optimizer = loss_scaler = lr_sched = None
    else:
        if args.lr is None:
            if rank == 0:
                print('using relative lr (per 256 samples):', args.blr, flush=True)
            args.lr = args.blr * args.batch_size * world / 256.0
            if rank == 0:
                print('effective lr:', args.lr, flush=True)
        else:
            if rank == 0:
                print('using absolute lr:', args.lr, flush=True)

        params_with_decay, params_without_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (params_without_decay if n.endswith(".bias") else params_with_decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {'params': params_with_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': params_without_decay, 'lr': args.lr, 'weight_decay': 0.0},
            ],
        )
        if rank == 0:
            print(optimizer, flush=True)

        loss_scaler = torch.cuda.amp.GradScaler()

        def lr_func(step):
            # avoid div by zero if someone runs eval_only
            steps_per_epoch = max(1, len(dataloader_train))
            epoch = step / steps_per_epoch
            min_lr_ratio = 0.01  # min_lr is 1e-2 times the base lr
            if epoch < args.warmup_epochs:
                return min_lr_ratio + (1.0 - min_lr_ratio) * (epoch / args.warmup_epochs)
            prog = (epoch - args.warmup_epochs) / max(1e-8, (args.epochs - args.warmup_epochs))
            return min_lr_ratio + (1.0 - min_lr_ratio) * (0.5 + 0.5 * math.cos(math.pi * prog))

        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    # --- Eval fn ---
    def evaluate(log_stats=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        num_classes = DATASETS[args.dataset]['NUM_CLASSES']
        device = next(model.parameters()).device
        if args.eval_only:
            class_correct = torch.zeros(num_classes, device=device, dtype=torch.long)
            class_total = torch.zeros(num_classes, device=device, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            for data, labels in metric_logger.log_every(dataloader_val, 100, header):
                data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                B, V = data.size(0), data.size(1)
                data = data.flatten(0, 1)
                with torch.cuda.amp.autocast():
                    logits = model.module(data)  # avoid no_sync in eval; use module directly
                    scores = logits.softmax(dim=-1).view(B, V, -1).mean(dim=1)
                preds = scores.topk(1, dim=1)[1].squeeze(1)
                acc1 = (preds == labels).float().mean().item() * 100
                acc5 = (scores.topk(5, dim=1)[1] == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
                metric_logger.meters['acc1'].update(acc1, n=scores.size(0))
                metric_logger.meters['acc5'].update(acc5, n=scores.size(0))
                if args.eval_only:
                    for c in range(num_classes):
                        mask = (labels == c)
                        class_total[c] += mask.sum()
                        class_correct[c] += ((preds == labels) & mask).sum()
        if args.eval_only:
            dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_total, op=dist.ReduceOp.SUM)
        metric_logger.synchronize_between_processes()
        if rank == 0:
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
                  .format(top1=metric_logger.acc1, top5=metric_logger.acc5), flush=True)
            if args.eval_only:
                # Per-class accuracy (eval-only mode)
                class_names = DIVING48_CLASS_NAMES if args.dataset == 'diving48' and num_classes == len(DIVING48_CLASS_NAMES) else [f'Class_{c}' for c in range(num_classes)]
                print('\n=== Per-class accuracy ===', flush=True)
                per_class_acc = []
                for c in range(num_classes):
                    total_c = class_total[c].item()
                    correct_c = class_correct[c].item()
                    acc_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
                    per_class_acc.append(acc_c)
                    print(f'  {c:2d} {class_names[c]:40s}  acc={acc_c:6.2f}%  ({correct_c}/{total_c})', flush=True)
                classes_with_samples = [c for c in range(num_classes) if class_total[c].item() > 0]
                if classes_with_samples:
                    mean_per_class = np.mean([per_class_acc[c] for c in classes_with_samples])
                    print(f'\n  Mean per-class accuracy (over classes with samples): {mean_per_class:.2f}%', flush=True)
                if args.save_dir:
                    per_class_path = os.path.join(args.save_dir, 'per_class_accuracy.txt')
                    with open(per_class_path, 'w') as f:
                        f.write('class_index\tclass_name\tcorrect\ttotal\taccuracy_pct\n')
                        for c in range(num_classes):
                            total_c = class_total[c].item()
                            correct_c = class_correct[c].item()
                            acc_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
                            f.write(f'{c}\t{class_names[c]}\t{correct_c}\t{total_c}\t{acc_c:.2f}\n')
                    print(f'  Per-class accuracy saved to {per_class_path}', flush=True)
        if log_stats is not None and rank == 0:
            log_stats.update({'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()})
            

    def visualize_attention_overlays(model, dataloader, args):
        if len(dataloader) == 0:
            print("Validation dataloader empty; skipping attention visualization.", flush=True)
            return

        max_batches = args.visualize_attn_batches if getattr(args, 'visualize_attn_batches', 1) > 0 else len(dataloader)
        data_iter = iter(dataloader)
        device = next(model.parameters()).device
        base_model = model.module if hasattr(model, 'module') else model
        was_training = model.training
        model.eval()

        base_dir = args.save_dir or os.getcwd()
        output_dir = os.path.join(base_dir, 'attention_overlays')
        os.makedirs(output_dir, exist_ok=True)

        for batch_idx in range(max_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                print("Reached end of validation loader before capturing all requested batches.", flush=True)
                break

            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            frames_for_plot = data[0, 0].cpu().clone()  # first sample, first view
            flattened = data.to(device, non_blocking=True).flatten(0, 1)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _, attn_maps = base_model(flattened, attn_weights=True)

            if not attn_maps:
                print(f"Batch {batch_idx}: Attention maps were not returned; skipping overlay.", flush=True)
                continue

            attn_tensor = attn_maps[-1].detach().cpu()
            num_frames = frames_for_plot.size(1)
            if num_frames <= 0:
                print("Empty frame sequence; skipping overlay.", flush=True)
                continue
            if attn_tensor.shape[0] < num_frames:
                print(f"Only {attn_tensor.shape[0]} attention entries for {num_frames} frames; skipping overlay.", flush=True)
                continue

            attn_tensor = attn_tensor[:num_frames]
            num_patches = attn_tensor.shape[-1] - 1
            grid_size = int(math.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                print("Unexpected patch grid size when visualizing attention; skipping.", flush=True)
                continue

            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=frames_for_plot.dtype).view(-1, 1, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=frames_for_plot.dtype).view(-1, 1, 1, 1)
            frames_for_plot = frames_for_plot * std + mean
            frames_for_plot = frames_for_plot.clamp(0.0, 1.0)

            T = frames_for_plot.size(1)
            H, W = frames_for_plot.size(2), frames_for_plot.size(3)

            for t in range(T):
                cls_attn = attn_tensor[t][:, 0, 1:].mean(0)
                cls_map = cls_attn.view(grid_size, grid_size)
                attn_upsampled = F.interpolate(
                    cls_map.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

                attn_np = attn_upsampled.cpu().numpy()
                frame = frames_for_plot[:, t].permute(1, 2, 0).numpy()
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                ax.imshow(frame)
                ax.imshow(attn_np, cmap='jet', alpha=0.5, extent=(0, W, H, 0))
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout(pad=0)
                fig.savefig(os.path.join(output_dir, f"attention_overlay_batch_{batch_idx:03d}_frame_{t:02d}.png"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            print(f"Saved attention overlays for batch {batch_idx} to {output_dir}", flush=True)

        if was_training:
            model.train()

    # --- Resume / Pretrain ---
    start_epoch = load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler)

    if args.eval_only:
        evaluate()
        return

    # --- Train ---
    for epoch in range(start_epoch, args.epochs):
        # make sure each replica shuffles differently
        dataloader_train.sampler.set_epoch(epoch)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch}]'

        model.train()
        for step, (data, labels) in enumerate(metric_logger.log_every(dataloader_train, args.print_freq, header)):
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(data)
                acc1 = (logits.topk(1, dim=1)[1] == labels.view(-1, 1)).float().mean().item() * 100
                #acc5 = (logits.topk(5, dim=1)[1] == labels.view(-1, 1)).float().mean().item() * 100
                acc5 = (logits.topk(5, dim=1)[1] == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
                loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)       # actually updates the params
            loss_scaler.update()              # updates the scaler
            lr_sched.step()                   # step AFTER optimizer

            metric_logger.update(
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr'],
                acc1=acc1, acc5=acc5,
            )

        if rank == 0:
            print('Averaged stats:', metric_logger, flush=True)
            log_stats = {'train_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
        else:
            log_stats = {}

        save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler)

        if ((epoch + 1) % args.eval_freq == 0) or ((epoch + 1) == args.epochs):
            evaluate(log_stats)

        if args.save_dir is not None and rank == 0:
            n_total_params = sum(p.numel() for p in model_without_ddp.parameters())
            log_stats['epoch'] = epoch
            log_stats['n_trainable_params'] = n_trainable_params
            log_stats['n_total_params'] = n_total_params
            with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    main()