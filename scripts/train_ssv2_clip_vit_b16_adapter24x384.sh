#!/usr/bin/env bash
#SBATCH --job-name=ssv2_ddp
#SBATCH --output=job_name%j.%N.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4        # 4 GPUs per node = 8 tasks total
#SBATCH --cpus-per-task=16         # adjust if needed
#SBATCH --gres=gpu:4
#SBATCH --time=00-07:00:00
#SBATCH --mem=0
#SBATCH --account=hk-project-pai00116
#SBATCH --partition=accelerated    # HoreKa GPU partition

echo "=== Activating environment ==="
source /home/hk-project-pai00116/id_glh1237/st-adapter/st_adapter_env/bin/activate

# -------------------------
# Distributed environment setup
# -------------------------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Debug Info ==="
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "Node list: $SLURM_JOB_NODELIST"

# -------------------------
# Step 1: Run quick DDP sanity test
# -------------------------
echo "=== Running DDP sanity test ==="

cat << 'EOF' > ddp_sanity.py
import os, torch, torch.distributed as dist

def main():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size
    )

    x = torch.tensor([rank+1], device="cuda")
    dist.all_reduce(x)
    print(f"[DDP-TEST] host={os.uname()[1]} rank={rank} local_rank={local_rank} "
          f"world={world_size} allreduce={x.item()} expected={world_size*(world_size+1)//2}",
          flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
EOF

srun python ddp_sanity.py || { echo "❌ DDP sanity test failed, aborting job."; exit 1; }

echo "✅ DDP Sanity Test Passed, launching training..."

# -------------------------
# Step 2: Launch actual training
# -------------------------
srun python main.py \
  --model clip_vit_base_patch16_adapter24x384_ms \
  --save_dir output_dir/ssv2/clip_vit_base_patch16_adapter24x384_ms\
    --auto_resume --auto_remove \
    --dataset ssv2 \
    --blr 0.002 \
    --num_frames 32 \
    --sampling_rate 0 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --batch_size 8 \
    --epochs 60 \
    --warmup_epochs 2 \
    --eval_freq 1 
