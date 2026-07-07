# profile_grid_v2

Extended robustness measurement matrix for the vLLM offload timing project.

Purpose:
- New canonical measurement matrix for workload-shape robustness.
- Does not overwrite old Porto runs.
- Varies model, CPU offload, concurrency, input length, and output length.

Planned matrix:
- models: llama, qwen
- offload_gb: 0, 2, 4, 8, 12, 16
- concurrency: 1, 2, 4, 8, 12, 16
- input_len: 128, 256, 512
- output_len: 32, 64, 128
- runs_per_cell: 5
- requests_per_run: 20

Output root:
new/runs/profile_grid_v2/
