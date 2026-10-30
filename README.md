# Context16 Concurrent String Index

这是一个**只包含最新一版 Context16 字符串索引**的独立仓库：原索引 `index.hpp` 保持不动，新增一个并发包装层，使同一个实例支持**多读者 + 多写者**。

> 只取 `czr112/string-learned-index` 的 `02_Code_and_Experiments/validation_e25_20260918/index.hpp`
> （SHA-256 `c8a1ae3c7373714842f724887a6f45eea7ce5ab7b6e9cdbd677ab87fed0e7265`，约 6,900 行），
> 不包含 BM、F5/F6、E3–E24 等历史版本，也不包含评测框架。

## 1. 并发模型

- **多读者**：`find` / `scan` / `all` / `size` / `memory` 可以同时调用，读者不拿写锁。
- **多写者**：`insert` / `update` / `erase` 可以同时调用；不同 shard 的写并行，同一 shard 的写串行。
- **读写并发**：写者发布新的不可变快照，读者读旧/新快照；`scan` 用 generation 检查，避免跨 shard 混合两代数据。
- **单实例**：`bulk_load` 之后，所有线程共享同一个 `ConcurrentIndex`。

实现方式：

```text
ConcurrentIndex
 ├─ shard 0   (空 key)
 ├─ shard 1   (首字节 0x00)
 ├─ ...
 └─ shard 256 (首字节 0xff)
       ├─ std::map<string, uint64>            写者维护的有序 key/value
       └─ shared_ptr<const validation::Index> 读者看到的不可变快照
```

写者：锁该 shard → 复制 map → 改 → 构建新的 `validation::Index` → 原子发布 → 旧快照由 `shared_ptr` 引用计数回收。
读者：`atomic_load(shared_ptr)` → 在该不可变快照上调用原索引的 `find` / `scan`。

## 2. 与 ART-OLC 的对应关系

| ART-OLC 概念 | 本实现 |
|---|---|
| 节点版本字 / lock / obsolete | 不需要：快照不可变，读者不会看到半更新 |
| `readLockOrRestart` / `checkOrRestart` | `std::atomic_load(shared_ptr, acquire)`；`find` 不需要 restart |
| `writeLock` / 锁耦合 | 每 shard 一个锁；没有父子节点锁 |
| obsolete + EBR | 旧快照由 `shared_ptr` 引用计数回收 |
| 多写者冲突 | 不同 shard 并行；同一 shard 串行 |
| 读者重试 | `scan` / `memory` 检测到写重叠时重试 |

这不是逐节点 OLC，而是**粗粒度 RCU/COW**。原 `Region` 的 `storage_`、`overflow_owners_`、`dirs_` 和 `Index::nodes_` 都是原地修改的，逐节点加版本字等于重写 `Region`（约 4,500 行）；包装层选择“一个 shard 的完整 `Index`”作为发布单元，保留 OLC 的三条核心契约：读者不写共享状态、写者发布新版本、旧版本安全后回收。

## 3. 目录

```text
context16-concurrent-index/
  CMakeLists.txt
  README.md
  NOTICE.md
  include/
    index.hpp                 # 最新 Context16 索引，未改动
    switches.hpp              # 冻结开关/基线配置
    model/                    # LITS HPT / BlockLinearRank / EqualDepthDir 后端
    vendor/lits_model_safe/   # LITS 模型依赖
    concurrent/
      compat.hpp              # POSIX std::mutex/thread；Windows spinlock/CreateThread
      concurrent_index.hpp    # 并发包装层
  tests/
    concurrent_index_test.cpp # 多线程测试
```

## 4. 构建与测试

Linux（目标环境）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

或者直接用 g++：

```bash
g++ -std=c++17 -O2 -pthread \
  -I include -I include/concurrent \
  tests/concurrent_index_test.cpp -o build/concurrent_index_test
./build/concurrent_index_test
```

Windows + MinGW 8.1（win32 线程模型，走 `compat.hpp` 的 spinlock/CreateThread）：

```powershell
g++ -std=c++17 -O2 -I include -I include/concurrent `
  tests/concurrent_index_test.cpp -o build/concurrent_index_test.exe
.\build\concurrent_index_test.exe
```

预期输出：

```text
PASSED in ~270 ms
```

测试覆盖：空索引/单键/空键/NUL/0xff、顺序增删改查、`scan` 边界、8 个写者写不同 shard、4 个写者写同一 shard、8 个读者并发 `find/scan/size/memory`、写完后与 `std::map` 参照全量对比。

## 5. 用法

```cpp
#include "concurrent/concurrent_index.hpp"

concurrent_index::ConcurrentIndex index;

std::vector<validation::Pair> rows = {
    {"a", 1}, {"ab", 2}, {"b", 3},
};
index.bulk_load(rows);                 // 启动线程之前调用一次

auto value = index.find("ab");         // std::optional<uint64_t>
auto page = index.scan("a", 32);       // 至多 32 条，按字节序
index.insert("ac", 4);                 // 已存在返回 false，保留旧值
index.update("a", 7);                  // 缺失返回 false
index.erase("b");                      // 缺失返回 false
```

`bulk_load` 必须在任何读写线程启动前完成；析构前必须 join 所有线程。

## 6. 限制

1. **写放大**：每次写复制该 shard 的全部 key/value，并重建整个 shard 的 `validation::Index`；写代价 `O(shard_size)`，不是 `O(log n)`。
2. **同 shard 写串行**：不同 shard 的写并行，同一 shard 的写由一把锁串行；没有实现 shard 内 per-region 多写者并行。
3. **首字节倾斜**：所有 key 都以 `'h'` 开头（URL）时，`0x68` shard 会很大；可按首 2 字节或前缀树再分片。
4. **空间开销**：每个非空 shard 都有独立的模型/区域/块和一份 map；小 shard 会重复支付模型开销。
5. **仅 Context16**：构造函数拒绝 LITS/BlockLinearRank/EqualDepthDir 后端；`compact_region_routing` 必须关闭（否则 `find` 会惰性写 mutable 路由缓存）。
6. **不要开诊断宏**：`E25_DIAG` / `E24_DIAG` / `E30_TIME` 等会写全局计数器/计时器，不是线程安全的。
7. **不持久化、不提供 `range_sum`**：与原索引一致；`range_sum` 需要另接。
8. **测试是压力/冒烟测试**：不能证明任意调度下无 data race；正式使用建议在目标平台跑 ThreadSanitizer 或做线性化检查。

## 7. 与 `learned-index` benchmark 的关系

本仓库是**索引本体**，不包含 `string-index/v1` 适配器；适配器在原来的研究仓库：

```text
07_Eval_Integration/adapters/our_index_concurrent/
```

它声明 `concurrent_find = true`，可直接接 `zsh20070718/learned-index` 的 `--threads`。benchmark 的 mixed 负载是单线程的，多写者能力用本仓库的 `concurrent_index_test` 验证。

## 8. 来源与许可

- 上游：`czr112/string-learned-index`，`02_Code_and_Experiments/validation_e25_20260918/index.hpp`，SHA-256 `c8a1ae3c...`。
- 上游仓库**没有 LICENSE 文件**；本仓库只是本地研究副本，不声称拥有上游代码的再分发权。对外发布前请先向上游作者确认许可。
- `include/concurrent/` 下的包装层和 `tests/` 是新代码；本仓库未声明许可证，使用者需自行决定。
