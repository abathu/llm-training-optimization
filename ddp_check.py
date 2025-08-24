import os, torch, torch.distributed as dist
from deepspeed import comm as ds_comm

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    x = torch.tensor([local_rank], device=f"cuda:{local_rank}")
    dist.all_reduce(x)
    print(f"[rank{dist.get_rank()}] cuda_count={torch.cuda.device_count()} local_rank={local_rank} tensor={x.item()}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()