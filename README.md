# Context16 String Index: v1 snapshot + v2 Region-OLC

这个仓库包含**两个并存版本**的 Context16 字符串索引并发实现，基线都是 `czr112/string-learned-index` 的
`02_Code_and_Experiments/validation_e25_20260918/index.hpp`
（SHA-256 `c8a1ae3c7373714842f724887a6f45eea7ce5ab7b6e9cdbd677ab87fed0e7265`，未修改）。

| 版本 | 位置 | 并发模型 | 写代价 | tag |
|---|---|---|---|---|
| **v1** | 根目录 `include/concurrent/` | shard 级不可变快照（RCU/COW） | `O(shard_size)` | `v1.0.0` |
| **v2** | `v2-olc/` | Region/Group 级 OLC（版本字 + lock/obsolete + 原地 value 更新） | `O(group_size)`，value 更新 `O(1)` | `v2.0.0-olc` |

v2 只新增了 `v2-olc/` 目录，**没有覆盖 v1**。详细设计、测试和取舍见 [WORKLOG.md](WORKLOG.md) 与 [v2-olc/README.md](v2-olc/README.md)。

---

## 快速开始

### v2（Region-OLC，推荐用于写较多的场景）

```bash
g++ -std=c++17 -O2 -pthread \
  -I v2-olc/include \
  v2-olc/tests/olc_index_test.cpp -o build/v2_test
./build/v2_test
```

Windows + MinGW 8.1：

```powershell
g++ -std=c++17 -O2 -I v2-olc/include `
  v2-olc/tests/olc_index_test.cpp -o build/v2_test.exe
.\build\v2_test.exe
```

预期输出：

```text
PASSED in ~15 ms
```

### v1（shard 快照，读多写少）

```bash
g++ -std=c++17 -O2 -pthread \
  -I include -I include/concurrent \
  tests/concurrent_index_test.cpp -o build/v1_test
./build/v1_test
```

预期输出：

```text
PASSED in ~270 ms
```

### CMake

```bash
# v1
cmake -S . -B build/v1 -DCMAKE_BUILD_TYPE=Release
cmake --build build/v1 -j
ctest --test-dir build/v1 --output-on-failure

# v2
cmake -S v2-olc -B build/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/v2 -j
ctest --test-dir build/v2 --output-on-failure
```

---

## v2：Region/Group 级 OLC

### 结构

```text
OlcIndex
 ├─ 257 fixed Regions（0 = 空 key，1..256 = 首字节 0x00..0xff）
 │    ├─ prefix（首字节）
 │    ├─ Context16 Model（16×258，Q=2^47，MASS=2^16）
 │    └─ 固定 Group 槽数组（每槽一个 Version）
 │         └─ Group（不可变、按 suffix 排序的 Record 数组；value 是 atomic）
 └─ 全局 active writers / generation（仅 scan）
```

### OLC 协议

每个 Region 和每个 Group 槽都有 ART-OLC 风格的 64 位版本字：

```text
bit 63..62 type
bit 61..2  version
bit 1      lock
bit 0      obsolete
```

- 读者：`readLock` → 读 Group/Region → `check`；版本变化、加锁或 obsolete 就 restart。
- 写者：`writeLockOrRestart`（CAS `v -> v+2`）→ 修改 → `unlock`（`+2`）。
- `update`：持 Group 槽锁，`Record::value.store(...)`，**原地更新**。
- `insert`/`erase`：构建新 Group，CAS 替换槽指针，旧 Group `mark_obsolete()`，读者冲突 restart。
- Region 有 `version_` 变化计数器；任何写 `bump()`，`scan` 检测写重叠。
- 跨 Region `scan` 另有 `active_writers_` + `generation_`，避免混合两代。

### 与 ART-OLC 的关系

| ART-OLC | v2 |
|---|---|
| 节点版本字 / lock / obsolete | 有（Region + Group 槽） |
| 读者乐观校验 | 有 |
| 写者节点锁 | 有（Group 槽） |
| 结构变更新节点 + CAS 父指针 | 有（Group 替换） |
| value 原地更新 | 有（atomic） |
| EBR | **未实现**，用 `shared_ptr` 引用计数回收 |
| 范围扫描 | 逐 Region/Group 校验，冲突重试 |

### v2 API

```cpp
#include "olc/index.hpp"

olc::OlcIndex index;
std::vector<olc::Pair> rows = {{"a", 1}, {"ab", 2}, {"b", 3}};
index.bulk_load(rows);              // 启动线程之前调用一次

auto value = index.find("ab");      // std::optional<uint64_t>
auto page = index.scan("a", 32);    // 至多 32 条
index.insert("ac", 4);              // 已存在返回 false
index.update("a", 7);               // 缺失返回 false
index.erase("b");                   // 缺失返回 false
```

### v2 限制

1. **Group 不分裂**：一个 Group 可以增长，insert/erase 的 COW 成本 `O(group_size)`；热点 Group 会变慢。
2. **首字节分区**：URL 都以 `'h'` 开头时，`0x68` Region 会成为热点；可按首 2 字节或自适应前缀再分。
3. **回收不是 EBR**：用 `shared_ptr` 引用计数；旧 Group 在最后一个读者释放后回收。
4. **`bulk_load` 单线程**：257 个 Region 顺序构建。
5. **新的 Region 存储**：v2 用 Group 有序数组重写了 Region，保留了 Context16 Model/prefix 路由，但**没有沿用**原 block/overflow/demand capacity 布局。
6. **不持久化、不提供 `range_sum`**：与原索引一致。

---

## v1：shard 级不可变快照

### 结构

```text
ConcurrentIndex
 ├─ shard 0   (空 key)
 ├─ shard 1   (首字节 0x00)
 ├─ ...
 └─ shard 256 (首字节 0xff)
       ├─ std::map<string, uint64>            写者维护的有序 key/value
       └─ shared_ptr<const validation::Index> 读者看到的不可变快照
```

- 写者：锁该 shard → 复制 map → 改 → 构建新的 `validation::Index` → 原子发布。
- 读者：`atomic_load(shared_ptr)` → 在不可变快照上调用原索引的 `find` / `scan`。
- 不同 shard 的写并行；同一 shard 的写串行。
- `scan` 用跨 shard 的 generation 检查，写重叠时重试。
- 旧 shard 快照由 `shared_ptr` 引用计数回收。

### v1 API

```cpp
#include "concurrent/concurrent_index.hpp"

concurrent_index::ConcurrentIndex index;
std::vector<validation::Pair> rows = {{"a", 1}, {"ab", 2}, {"b", 3}};
index.bulk_load(rows);

auto value = index.find("ab");
auto page = index.scan("a", 32);
index.insert("ac", 4);
index.update("a", 7);
index.erase("b");
```

### v1 限制

1. **写放大**：每次写复制该 shard 的全部 key/value，并重建整个 shard 的 `validation::Index`；写代价 `O(shard_size)`。
2. **同 shard 写串行**：不同 shard 并行，同一 shard 由一把锁串行。
3. **首字节倾斜**：URL 类 key 的 `0x68` shard 会很大。
4. **空间开销**：每个非空 shard 有独立的模型/区域/块和一份 map。
5. **仅 Context16**：`compact_region_routing` 必须关闭，否则 `find` 会写 mutable 路由缓存。
6. **不要开诊断宏**：`E25_DIAG` / `E24_DIAG` / `E30_TIME` 等会写全局计数器/计时器。

---

## 目录

```text
context16-concurrent-index/
  README.md                       本文件
  WORKLOG.md                      v1/v2 工作日志、区别、测试证据
  NOTICE.md                       上游来源与许可说明
  CMakeLists.txt                  v1 构建
  include/
    index.hpp                     Context16 原索引，未改动
    switches.hpp                  冻结开关/基线配置
    model/                        LITS HPT / BlockLinearRank / EqualDepthDir
    vendor/lits_model_safe/       LITS 模型依赖
    concurrent/
      compat.hpp                  POSIX std::mutex/thread；Windows spinlock/CreateThread
      concurrent_index.hpp        v1 shard 快照包装层
  tests/
    concurrent_index_test.cpp     v1 多线程测试
  v2-olc/
    CMakeLists.txt                v2 构建
    README.md                     v2 设计说明
    include/olc/
      version.hpp                 64 位版本字
      model.hpp                   Context16 模型
      group.hpp                   排序 Group；value atomic
      region.hpp                  Region OLC 读写/scan
      index.hpp                   257 Region、全局 size、scan generation
      compat.hpp                  测试用线程/锁兼容层
    tests/
      olc_index_test.cpp          基础/顺序/并发/同 shard
      olc_random_test.cpp         4 seed × 2500 随机差分
      olc_value_test.cpp          8 writer + 8 reader 值一致性
      olc_contention_test.cpp     单 Group 4 writer + 4 reader
```

---

## 测试结果

本机 MinGW 8.1（Windows）用 g++ 直接编译运行：

| 测试 | 结果 |
|---|---|
| v1 `tests/concurrent_index_test.cpp` | PASSED，约 265–276 ms；重复 5 次 0 失败 |
| v2 `v2-olc/tests/olc_index_test.cpp` | PASSED，约 13–17 ms；重复 20 次 0 失败 |
| v2 `olc_random_test.cpp` | 4 seed × 2500 操作，0 mismatch |
| v2 `olc_value_test.cpp` | 8 writer + 8 reader，10 次 0 失败，最终 8000 键 |
| v2 `olc_contention_test.cpp` | 单 Group 4 writer + 4 reader，20 次 0 失败 |
| v2 `-DNDEBUG` Release | 通过 |

说明：本机没有 cmake/WSL，Linux 的 `std::mutex/std::thread` 路径和 `ctest` 没有实际执行；CMake 文件已提供。
这些是差分/压力测试，不是线性化证明；正式使用建议在 Linux 上跑 ThreadSanitizer。

---

## v1 与 v2 的选择

| 场景 | 推荐 |
|---|---|
| 读远多于写，写低频/批量 | v1 或 v2 都可；v1 实现更简单 |
| 写较多，想减少写放大 | **v2**（只 COW 一个 Group） |
| 同一首字节热点严重 | v1/v2 都会受影响；需要更细分片 |
| 要求严格 ART-OLC（含 EBR） | 当前 v2 仍是 `shared_ptr` 回收；需要继续升级 |
| 构建后只读 | 直接冻结单实例即可，不需要并发层 |

---

## 与 `learned-index` benchmark 的关系

本仓库是**索引本体**，不包含 `string-index/v1` 适配器。v1 的适配器在原来的研究仓库：

```text
07_Eval_Integration/adapters/our_index_concurrent/
```

v2 目前只有索引本体和测试；如果要接 benchmark，需要按 `string-index/v1` 写一个薄适配器，把
`olc::OlcIndex` 的 `find/insert/update/erase/scan/memory` 转发出去。

---

## 来源与许可

- 上游：`czr112/string-learned-index`，`02_Code_and_Experiments/validation_e25_20260918/index.hpp`，SHA-256 `c8a1ae3c...`。
- 上游仓库**没有 LICENSE 文件**；本仓库只是本地研究副本，不声称拥有上游代码的再分发权。对外发布前请先向上游作者确认许可。
- `include/concurrent/`、`v2-olc/` 和测试是新代码；本仓库未声明许可证，使用者需自行决定。

---

## 相关链接

- [工作日志](WORKLOG.md)
- [v2 设计说明](v2-olc/README.md)
- [v2 代码目录](v2-olc)
- [v1 代码目录](include/concurrent)
- [v2 tag](https://github.com/panzirui0104-lab/learned-index/tree/v2.0.0-olc)
- [v1 tag](https://github.com/panzirui0104-lab/learned-index/tree/v1.0.0)
