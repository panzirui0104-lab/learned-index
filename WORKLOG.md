# 工作日志：Context16 并发索引 v1 → v2

记录日期：2026-11-01  
仓库：`panzirui0104-lab/learned-index`  
基线：`czr112/string-learned-index` 的 `02_Code_and_Experiments/validation_e25_20260918/index.hpp`（SHA-256 `c8a1ae3c...`，未修改）

**当前状态（2026-11-01）**

- v2 代码提交：`04a49de`（工作日志更新是 `main` 上后续的 docs-only 提交）
- v2 tag：`v2.0.0-olc`
- v1 tag：`v1.0.0`（`4bc0a64`）
- v1 在根目录保留；v2 在 `v2-olc/`；两个版本并存
- 已通过 `git diff 4bc0a64..04a49de -- include tests` 确认 v1 代码零改动

---

## 1. 两个版本并存

| 版本 | 位置 | 并发模型 | 状态 |
|---|---|---|---|
| **v1** | 根目录 `include/`、`tests/` | shard 级不可变快照（RCU/COW） | 已发布在 `main`，保留不动 |
| **v2** | `v2-olc/` | Region/Group 级 OLC（版本字 + lock/obsolete + 原地 value 更新） | 新增，未覆盖 v1 |

v2 只新增 `v2-olc/` 目录，没有修改 v1 的 `include/index.hpp`、`include/concurrent/` 或 `tests/`。

查看两个版本：

```bash
# v1：根目录
git checkout v1.0.0
ls include/concurrent

# v2：v2-olc/
git checkout v2.0.0-olc
ls v2-olc/include/olc
```

GitHub 上也可以直接打开 [v2-olc](https://github.com/panzirui0104-lab/learned-index/tree/main/v2-olc) 和 [WORKLOG](https://github.com/panzirui0104-lab/learned-index/blob/main/WORKLOG.md)。

---

## 2. v1 做了什么

**结构**

```text
ConcurrentIndex
 ├─ 257 shards（按首字节）
 │    ├─ std::map<string, uint64>            写者维护
 │    └─ shared_ptr<const validation::Index> 读者快照
 └─ 全局 active writers / generation
```

**特点**

- 读者 `atomic_load(shared_ptr)` 读不可变 `validation::Index` 快照。
- 写者复制整个 shard 的 map，重建整个 `validation::Index`，原子发布。
- 不同 shard 的写并行；同 shard 写串行。
- `scan` 用 generation 检查，写重叠时重试。
- 旧 shard 快照由 `shared_ptr` 引用计数回收。

**代价**

- 每次写 `O(shard_size)`；同一 shard 的写完全串行。
- 首字节热点 shard（URL 都以 `'h'` 开头）会成为瓶颈。
- 读者每次 `find` 有 `shared_ptr` 原子 load + 引用计数开销。

**测试**

- `tests/concurrent_index_test.cpp`：PASSED，约 265–276 ms；重复 5 次 0 失败。
- 随机差分：4 seed × 2500 操作，0 mismatch。
- 并发值一致性：8 writer + 8 reader，10 次 0 失败。

---

## 3. v2 做了什么

**结构**

```text
OlcIndex
 ├─ 257 fixed Regions（0 = 空 key，1..256 = 首字节）
 │    ├─ prefix（首字节）
 │    ├─ Context16 Model（16×258，Q=2^47，MASS=2^16）
 │    └─ Group 槽数组（每槽一个 Version）
 │         └─ Group（不可变排序 Record 数组；value 是 atomic）
 └─ 全局 active writers / generation（仅 scan）
```

**OLC 协议**

- 版本字布局：

```text
bit 63..62 type
bit 61..2  version
bit 1      lock
bit 0      obsolete
```

- 读者：`readLock` → 读 Group/Region → `check`；版本变化/加锁/obsolete 就 restart。
- 写者：`writeLockOrRestart`（CAS `v -> v+2`），改完 `unlock`（`+2`）。
- `insert`/`erase`：构建新 `Group`，CAS 替换槽指针，旧 Group `mark_obsolete()`，读者冲突 restart。
- `update`：持 Group 槽锁，`Record::value.store(...)`，真正原地更新。
- Region 有 `version_` 变化计数器；任何写 `bump()`，`scan` 检测写重叠。
- 跨 Region `scan` 另有 `active_writers_` + `generation_`，避免混合两代。

**代码文件**

| 文件 | 作用 |
|---|---|
| `v2-olc/include/olc/version.hpp` | 64 位版本字：version / lock / obsolete |
| `v2-olc/include/olc/model.hpp` | Context16 模型（16×258，Q=2^47，MASS=2^16） |
| `v2-olc/include/olc/group.hpp` | 不可变排序 Group；value 是 atomic |
| `v2-olc/include/olc/region.hpp` | Region：prefix / model / Group 槽 / OLC 读写 / scan |
| `v2-olc/include/olc/index.hpp` | 257 个 Region、全局 size、scan generation |
| `v2-olc/include/olc/compat.hpp` | POSIX std::mutex/thread；Windows spinlock/CreateThread（测试用） |
| `v2-olc/tests/olc_index_test.cpp` | 基础/顺序/并发/同 shard 测试 |
| `v2-olc/tests/olc_random_test.cpp` | 4 seed × 2500 随机差分 |
| `v2-olc/tests/olc_value_test.cpp` | 并发值一致性（8 writer + 8 reader） |
| `v2-olc/tests/olc_contention_test.cpp` | 单 Group 争用（4 writer + 4 reader） |

**与 ART-OLC 的对应**

| ART-OLC | v2 |
|---|---|
| 节点版本字 / lock / obsolete | 有（Region + Group 槽） |
| 读者乐观校验 | 有 |
| 写者节点锁 | 有（Group 槽） |
| 结构变更新节点 + CAS 父指针 | 有（Group 替换） |
| value 原地更新 | 有（atomic） |
| EBR | **未实现**，用 `shared_ptr` 引用计数回收 |
| 范围扫描 | 逐 Region/Group 校验，冲突重试 |

**代价/限制**

- Group 不分裂；一个 Group 可以增长，insert/erase COW 成本 `O(group_size)`。
- 首字节分区仍有热点；可按首 2 字节或自适应前缀再分。
- `bulk_load` 单线程；257 个 Region 顺序构建。
- 回收不是 epoch EBR；是 `shared_ptr` 引用计数。

**测试**

- `v2-olc/tests/olc_index_test.cpp`：PASSED，约 13–17 ms；重复 20 次 0 失败。
- `v2-olc/tests/olc_random_test.cpp`：4 seed × 2500 操作，0 mismatch。
- `v2-olc/tests/olc_value_test.cpp`：8 writer + 8 reader，10 次 0 失败，最终 8000 键。
- `v2-olc/tests/olc_contention_test.cpp`：单 Group 4 writer + 4 reader，20 次 0 失败。
- `-DNDEBUG` Release：通过。

以上测试均在本机 MinGW 8.1（Windows）用 g++ 直接编译运行；Linux 的 `std::mutex/std::thread` 路径未在本机执行，CMake 文件已提供但本机没有 cmake，未跑 `ctest`。

---

## 4. v1 与 v2 的关键区别

| 维度 | v1 | v2 |
|---|---|---|
| 并发单元 | shard（257 个） | Region + Group 槽 |
| 读路径 | `atomic_load(shared_ptr)` + 原 `Index::find` | 版本字 `readLock` + Group 二分 + `check` |
| 写路径 | 复制 shard map + 重建 `validation::Index` | Group 槽锁；value 原地写；insert/erase 只 COW 一个 Group |
| 写代价 | `O(shard_size)` | `O(group_size)`（value 更新 O(1)） |
| 多写者 | 不同 shard 并行；同 shard 串行 | 不同 Group 并行；同 Group 串行 |
| 读者是否被写阻塞 | 否 | 否（冲突 restart） |
| 旧版本回收 | `shared_ptr` 引用计数 | `shared_ptr` 引用计数（仍是差异项） |
| 版本字 | 无 | 有（Region + Group 槽） |
| obsolete | 无 | 有（旧 Group） |
| scan 一致性 | 全局 generation | Region version + 全局 generation |
| 热点倾斜 | 首字节 shard | 首字节 Region / 热点 Group |
| 代码位置 | 根目录 `include/concurrent/` | `v2-olc/include/olc/` |
| 原 `index.hpp` | 未改 | 未改（v2 自己实现 Model/Region） |

---

## 5. 为什么 v2 不是直接改 v1

- v1 的 `validation::Index` / `Region` 是黑盒；要加版本字必须进入 `Region` 的 block 存储。
- 原 `Region` 的 `storage_`、`overflow_owners_`、`dirs_` 和 `Index::nodes_` 都是可变结构，直接加版本字会产生 UAF/撕裂，不是真 OLC。
- v2 因此重写了 Region 的存储：Group 是不可变有序数组，value 原子；结构变更只替换一个 Group。
- 代价是 v2 没有复用原 block layout/overflow/demand capacity；它是新的 Region 实现，但保留了 Context16 Model/prefix 的路由。

---

## 6. 下一步

`v2.0.0-olc` 已发布；以下是后续优化，不影响当前 v1/v2 的可用性：

1. 用 epoch-based reclamation 替换 `shared_ptr` 引用计数（补齐 ART-OLC 的回收方式）。
2. Group 分裂/合并：热点 Group 增长时拆成多个 Group，保持 `O(log)` 写。
3. 并行 `bulk_load`：257 个 Region 独立构建。
4. 首 2 字节/自适应分片，缓解首字节热点。
5. Linux + ThreadSanitizer 长跑；补线性化/历史检查器。
6. 如果要求严格 ART-OLC，下一步把 v2 的 Group 槽升级为 `Group*` + EBR，并给 Group 内部加 sub-block 版本。

---

## 7. 复现命令

```powershell
cd context16-concurrent-index

# v1
g++ -std=c++17 -O2 -I include -I include/concurrent `
  tests/concurrent_index_test.cpp -o build/v1_test.exe
.\build\v1_test.exe

# v2（四个测试）
$inc = "v2-olc/include"
$tests = "olc_index_test","olc_random_test","olc_value_test","olc_contention_test"
foreach ($t in $tests) {
  g++ -std=c++17 -O2 -I $inc "v2-olc/tests/$t.cpp" -o "build/$t.exe"
  if ($LASTEXITCODE -ne 0) { throw "build failed: $t" }
  & ".\build\$t.exe"
  if ($LASTEXITCODE -ne 0) { throw "test failed: $t" }
}
```

Linux 上把 `-pthread` 加上，或使用 `v2-olc/CMakeLists.txt`：

```bash
cd v2-olc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## 8. 推送记录

- v1：`4bc0a64`，tag `v1.0.0`
- v2：`04a49de`，tag `v2.0.0-olc`
- 远端：`https://github.com/panzirui0104-lab/learned-index`
- v2 没有覆盖 v1；`include/`、`tests/`、根 `CMakeLists.txt` 的 v1 内容未改。
- 工作日志本次更新会作为 `main` 上的一个 docs-only 提交；v1/v2 代码不变。
