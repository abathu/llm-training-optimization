from accelerate import PartialState
s = PartialState()
print(f"[hello] local_rank={s.local_process_index} world_size={s.num_processes}")
