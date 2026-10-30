# Sources and licensing

## Upstream index

- Repository: https://github.com/czr112/string-learned-index
- Path: `02_Code_and_Experiments/validation_e25_20260918/index.hpp`
- SHA-256: `c8a1ae3c7373714842f724887a6f45eea7ce5ab7b6e9cdbd677ab87fed0e7265`
- Bundled dependencies: `include/model/` and `include/vendor/lits_model_safe/` are copied from the same upstream directory so the header compiles standalone.

The upstream repository has no `LICENSE` file. Public readability is not a redistribution license. This repository is a local research copy; do not redistribute the upstream code without permission from its author.

## New code

- `include/concurrent/compat.hpp`
- `include/concurrent/concurrent_index.hpp`
- `tests/concurrent_index_test.cpp`

These files implement the concurrent wrapper and its tests. No license is declared for them in this repository; the author/user should choose one before publishing.

## Not included

- The `learned-index` benchmark harness.
- BM (BlockModel), F5/F6, and the historical `validation_e*` versions.
- Datasets and experiment results.
