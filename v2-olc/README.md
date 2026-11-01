# Context16 Region-OLC v2

这是 `v1`（shard 级不可变快照）之后的 **Region 级 OLC** 版本。它不修改 `v1` 的 `include/`，所有新代码在 `v2-olc/` 下。

## 1. 结构

```text
OlcIndex
 ├─ 257 fixed Regions（0 = 空 key，1..256 = 首字节 0x00..0xff）
      └─ Region
           ├─ prefix（首字节）
           ├─ Context16 Model（16×258 表，Q=2^47，MASS=2^16）
           ├─ 固定 Group 槽数组（每个槽一个 Version）
           └─ Group（不可变、按 suffix 排序的 Record 数组；value 是 atomic）
```

- `Region` 的 `prefix` 只有**首字节**，所以同 shard 的新 key 一定匹配前缀，不需要分裂 Region。
- `Model` 把 suffix 映射到 `group_count_` 个组；`bucket` 单调，所以组顺序 = key 顺序。
- 每个 `Group` 的 key 数组不可变；只有 `Record::value` 是 `std::atomic<uint64_t>`，可以原地更新。
- `insert`/`erase` 在 Group 内产生新 `Group`（结构变更），`update` 原地写 value。
- Group 满不分裂：会继续增长并整体 COW；这是当前版本的已知限制。

## 2. OLC 协议

每个 `Region` 和每个 `Group` 槽都有 ART-OLC 风格的 64 位版本字：

```text
bit 63..62 type
bit 61..2  version
bit 1      lock
bit 0      obsolete
```

- 读者：`readLock` → 读 Group/Region → `check`；版本变化/加锁/obsolete 就 restart。
- 写者：`writeLockOrRestart`（CAS `v -> v+2`）→ 构建新 Group 或原地写 value → 发布 → `unlock`（`+2`）。
- 替换 Group 后对旧 Group `mark_obsolete()`；读者检查 obsolete，冲突时 restart。
- Region 还有一个 `version_` 变化计数器；任何写都会 `bump()`，`scan` 用它检测写重叠。
- 跨 Region 的 `scan` 另有 `active_writers_` + `generation_` 检查，避免混合两代数据。

## 3. 与 ART-OLC 的差异

| 项 | v2 | ART-OLC |
|---|---|---|
| 版本字 / lock / obsolete | 有 | 有 |
| 读者乐观校验 | 有 | 有 |
| 写者节点锁 | 有（Group 槽） | 有（节点） |
| 结构变更新节点 + CAS 父指针 | 有（Group 替换） | 有（节点 grow/shrink） |
| value 原地更新 | 有（atomic） | 有 |
| 回收 | `shared_ptr` 引用计数 | epoch-based reclamation |
| 范围扫描 | 逐 Region/Group 校验，冲突重试 | 逐节点校验/restart |
| 分区 | 257 个首字节 Region | 无人工分区 |

**唯一没有照搬的是 EBR**：本版用 `shared_ptr` 引用计数安全回收旧 Group；它不是 epoch，但回收安全性等价。  如果要求严格 ART-OLC 的 EBR，下一步可以把 `shared_ptr` 换成 `Group*` + `Epoch`。

## 4. 构建与测试

```bash
cd v2-olc
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
PASSED in ~15 ms
```

另外跑过：

- 随机差分：4 seed × 2500 操作，对 `std::map` 比较 `insert/update/erase/find/scan/all/size`，0 mismatch。
- 并发值一致性：8 writer + 8 reader，10 次运行 0 失败，最终 8000 键。
- 单 Group 争用：4 writer + 4 reader，20 次运行 0 失败。
- `-DNDEBUG` Release：通过。

## 5. API

```cpp
#include "olc/index.hpp"

olc::OlcIndex index;
std::vector<olc::Pair> rows = {{"a", 1}, {"ab", 2}, {"b", 3}};
index.bulk_load(rows);

auto v = index.find("ab");          // std::optional<uint64_t>
auto rows = index.scan("a", 32);    // 至多 32 条
index.insert("ac", 4);              // 已存在返回 false
index.update("a", 7);               // 缺失返回 false
index.erase("b");                   // 缺失返回 false
```

`bulk_load` 必须在读写线程启动前完成；`bulk_load` 会检查重复 key。

## 6. 限制

1. **Group 不分裂**：一个 Group 可以无限增长，insert/erase 的 COW 成本 `O(group_size)`；热点组会变慢。
2. **首字节分区**：所有 URL 以 `'h'` 开头时，`0x68` Region 会成为热点；可按首 2 字节或自适应前缀再分。
3. **回收不是 EBR**：用 `shared_ptr` 引用计数；旧 Group 在最后一个读者释放后回收。
4. **`bulk_load` 单线程**：257 个 Region 顺序构建；可以后续并行化。
5. **模型组数固定**：`min(4096, n/32)`；不支持组数动态调整。
6. **测试不是线性化证明**：只是差分/压力测试；正式使用建议在 Linux 上跑 ThreadSanitizer。

## 7. 与 v1 的关系

- `v1`：根目录 `include/concurrent/`，shard 级不可变 `validation::Index` 快照；写代价 `O(shard)`。
- `v2`：`v2-olc/`，Region/Group 级 OLC；value 原地更新，结构变更只 COW 一个 Group；写代价 `O(group)`。
- v2 没有覆盖 v1；两个版本可以并存。
