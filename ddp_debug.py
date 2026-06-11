import os, torch, torch.distributed as dist

def main():
    local_rank = int(os.environ["LOCAL_RANK"])  # set by torchrun
    torch.cuda.set_device(local_rank)

    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    x = torch.tensor([rank + 1], device="cuda", dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[DDP-TEST] rank={rank} local_rank={local_rank} gpu={torch.cuda.current_device()} "
          f"allreduce={x.item()} expected={world*(world+1)//2}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()