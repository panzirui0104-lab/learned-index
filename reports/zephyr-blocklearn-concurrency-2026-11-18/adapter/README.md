# Zephyr BlockLearn concurrent-read adapter

This is a thin `string-index/v1` adapter around
`Zephyr761/Learned-Index-Model` (`block-adaptive-delta-shortdir-first`).
It exists so the benchmark can test **shared-instance concurrent reads** without
modifying the upstream model.

## What it declares

- `concurrent_find = true` — valid only for static, read-only use after
  `bulk_load`.
- `find` forwards to `LearnedIndex::find`.
- `instrument != 0` is rejected: the upstream `find` updates mutable
  `Statistics` counters only when instrumentation is enabled.
- `insert`, `update`, `erase`, `scan`, `range_sum` forward for single-thread use;
  they are **not** declared concurrent-safe.
- `bulk_load` must finish before query threads start.

This is an adapter-level concurrency declaration, not an upstream guarantee.
The upstream descriptor currently says "Single-threaded".

## Build / run

From the benchmark root:

```bash
python3 -B scripts/benchmark.py \
  --dataset /path/to/keys.txt \
  --implementation /path/to/benchmark_adapter_zephyr_concurrent \
  --workload examples/workloads/uniform.json \
  --threads 1 2 4 \
  --output results/zephyr-concurrent
```

The adapter's CMake adds the Zephyr `src` directory and links
`blocklearn_adapter`. Set `ZEPHYR_SRC` if the model checkout is not at the
default sibling path:

```bash
cmake -DZEPHYR_SRC=/path/to/Learned-Index-Model/src ...
```

The adaptive model requires an independent calibration file; generate one with
`blocklearn_calibrate --output planner-costs.txt` and set
`BLOCKLEARN_COST_PROFILE` before running the benchmark.

See [the experimental report](../reports/zephyr-blocklearn-concurrency-2026-11-18.md)
for measured results and limitations.
