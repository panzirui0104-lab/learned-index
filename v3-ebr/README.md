# Context16 Region-OLC v3 (EBR)

这是 `v2`（Region/Group 级 OLC，`shared_ptr` 回收）之后的 **v3**：把 Group 的回收从
`shared_ptr` 引用计数换成 **epoch-based reclamation（EBR）**。v3 只新增 `v3-ebr/`，不修改 v1/v2。

## 1. 结构

```text
OlcIndex
 ├─ 257 fixed Regions（0 = 空 key，1..256 = 首字节 0x00..0xff）
 │    └─ Region
 │         ├─ prefix（首字节）
 │         ├─ Context16 Model（16×258，Q=2^47，MASS=2^16）
 │         ├─ 固定 Group 槽数组：std::atomic<Group*>
 │         └─ EpochReclaimer&（旧 Group 的延迟回收）
 └─ 全局 active writers / generation（仅 scan）
```

- `Group*` 是裸指针，不再用 `shared_ptr<const Group>`。
- `insert`/`erase`：把旧 `Group*` 交给 `EpochReclaimer::retire`；读者用 `Guard` 保护。
- `update`：仍是原地 `Record::value.store`。
- `Region` 析构时删除当前 Group；已退休的 Group 由 `OlcIndex` 析构或 `flush()` 回收。

## 2. OLC + EBR

每个 Region 和每个 Group 槽仍有 64 位版本字：

```text
bit 63..62 type
bit 61..2  version
bit 1      lock
bit 0      obsolete
```

- 读者：`readLock` → 读 Group/Region → `check`；版本变化、加锁或 obsolete 就 restart。
- 写者：`writeLockOrRestart`（CAS `v -> v+2`）→ 修改 → `unlock`（`+2`）。
- 旧 Group `mark_obsolete()` 后 `epoch.retire(old)`；读者持有 `EpochReclaimer::Guard` 时不会回收。
- `EpochReclaimer`：全局 epoch + reader 计数；`Guard` 进入与 `try_gc()` 共用一把 mutex，避免“检查 readers==0 后、删除前”新读者进入的窗口。
- `try_gc()` 在 `readers == 0` 时推进 epoch 并删除已退休 Group。
- 跨 Region `scan` 另有 `active_writers_` + `generation_`，避免混合两代。

### 与 ART-OLC 的对应

| ART-OLC | v3 |
|---|---|
| 节点版本字 / lock / obsolete | 有（Region + Group 槽） |
| 读者乐观校验 | 有 |
| 写者节点锁 | 有（Group 槽） |
| 结构变更新节点 + CAS 父指针 | 有（Group 替换） |
| value 原地更新 | 有（atomic） |
| EBR | **有**（全局 epoch + reader guard） |
| 范围扫描 | 逐 Region/Group 校验，冲突重试 |

v3 的 EBR 是**全局 epoch + reader 计数**版本，不是 ART-OLC 的 per-thread epoch list；
安全性等价，但回收粒度更粗（有活跃读者时不回收）。`EpochReclaimer` 也提供
`pending()` / `flush()` 便于测试和析构。

## 3. 与 v2 的区别

| 项 | v2 | v3 |
|---|---|---|
| Group 持有方式 | `shared_ptr<const Group>` | `std::atomic<Group*>` |
| 旧 Group 回收 | `shared_ptr` 引用计数 | `EpochReclaimer::retire` + `try_gc` |
| 读者保护 | 引用计数 | `EpochReclaimer::Guard` |
| 回收延迟 | 最后一个 `shared_ptr` 释放 | 没有活跃 reader 后的 epoch 推进 |
| 回收测试 | 无 | `olc_ebr_test.cpp` |
| 代码位置 | `v2-olc/` | `v3-ebr/` |

## 4. 构建与测试

```bash
cd v3-ebr
g++ -std=c++17 -O2 -pthread -I include tests/olc_index_test.cpp -o build/olc_index_test
./build/olc_index_test
```

Windows + MinGW 8.1：

```powershell
g++ -std=c++17 -O2 -I include tests/olc_index_test.cpp -o build/olc_index_test.exe
.\build\olc_index_test.exe
```

本机实测：

```text
PASSED in ~16 ms
```

另外跑过：

- `olc_random_test`：4 seed × 2500 操作，0 mismatch。
- `olc_value_test`：8 writer + 8 reader，10 次运行 0 失败，最终 8000 键。
- `olc_contention_test`：单 Group 4 writer + 4 reader，20 次运行 0 失败。
- `olc_ebr_test`：retire/try_gc 有无 reader guard 的回收行为。
- `-DNDEBUG` Release：通过。

## 5. API

```cpp
#include "olc/index.hpp"

olc::OlcIndex index;
std::vector<olc::Pair> rows = {{"a", 1}, {"ab", 2}, {"b", 3}};
index.bulk_load(rows);

auto v = index.find("ab");          // std::optional<uint64_t>
auto page = index.scan("a", 32);    // 至多 32 条
index.insert("ac", 4);              // 已存在返回 false
index.update("a", 7);               // 缺失返回 false
index.erase("b");                   // 缺失返回 false

index.reclaim();                    // 显式触发一次 try_gc
std::size_t pending = index.pending_retired();
```

`bulk_load` 必须在读写线程启动前完成；`bulk_load` 会检查重复 key。
`reclaim()` / `pending_retired()` 主要用于测试和诊断；正常写入也会调用 `try_gc()`。
析构 `OlcIndex` 前必须 join 所有读者/写者线程；EBR 不负责析构时仍活跃的线程。

## 6. 限制

1. **Group 不分裂**：一个 Group 可以增长，insert/erase 的 COW 成本 `O(group_size)`。
2. **首字节分区**：URL 都以 `'h'` 开头时，`0x68` Region 会成为热点。
3. **EBR 是全局 reader 计数版本**：有活跃读者时不回收；不是 per-thread epoch list。
4. **`bulk_load` 单线程**：257 个 Region 顺序构建。
5. **新的 Region 存储**：v3 用 Group 有序数组重写了 Region，保留 Context16 Model/prefix 路由，但没有沿用原 block/overflow/demand capacity 布局。
6. **不持久化、不提供 `range_sum`**。
7. **v2/v3 都使用 namespace `olc`**：同一翻译单元只能包含其中一个版本的头文件。

## 7. 版本关系

- v1：根目录 `include/concurrent/`，shard 级 `shared_ptr` 快照。
- v2：`v2-olc/`，Region/Group OLC，`shared_ptr` 回收。
- v3：`v3-ebr/`，与 v2 相同的 OLC 结构，Group 用 EBR 回收。
- 三个版本并存；v3 不覆盖 v1/v2。
