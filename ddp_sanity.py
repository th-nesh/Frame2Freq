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
