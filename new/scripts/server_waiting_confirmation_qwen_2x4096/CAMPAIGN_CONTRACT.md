# Qwen Server-WAITING Burst-Shape Control

Purpose: compare burst shape while keeping total burst input-token budget fixed.

Reference campaign:
- 4 burst requests × 2048 input tokens
- total input-token budget: 8192

New campaign:
- 2 burst requests × 4096 input tokens
- total input-token budget: 8192

Unchanged:
- Qwen2.5-7B-Instruct
- offload {0,12}
- server_max_num_seqs {4,8}
- trigger after 16 received decode tokens
- victim workload 20 × 256 input / 64 output
- burst output length 16
- four paired repeats per cell
- 32 episodes / 16 paired blocks
- max_num_batched_tokens = 2048
- original official seed and seed namespace, so victim prompts,
  block order and condition order remain matched to the reference campaign

Must change:
- separate design version
- separate campaign name
- separate schedule fingerprint
- separate default schedule/output directories
- burst_parallel_requests: 2
- burst_input_len: 4096
- tests, validation bounds and auditor expectations accordingly

The existing server_waiting_confirmation source and all previous results
must remain byte-identical.
