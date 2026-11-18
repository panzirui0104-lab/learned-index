# BlockLearn 学习索引并发能力实验报告

日期：2026-11-18  
Benchmark：`zsh20070718/learned-index` @ `003a55b`  
被测实现：`Zephyr761/Learned-Index-Model` @ `3894c767`（`clear1018`）  
适配器：`adapter/`（只读并发包装）

---

## 1. 摘要

本实验把最新版 BlockLearn 学习索引接入 `string-index/v1` benchmark，在 100,000 个字符串键、200,000 次静态点查、3 次重复的条件下，测量共享同一索引实例的 1/2/4/8 线程并发只读能力。

主要结果（自适应模型，uniform 访问）：

| 线程数 | 吞吐量（M ops/s） | 相对 1 线程加速比 | 并行效率 | 单线程 P50 | 单线程 P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.62 | 1.000 | 1.000 | 400 ns | 1000 ns |
| 2 | 12.40 | 1.874 | 0.937 | 400 ns | 900 ns |
| 4 | 25.24 | 3.814 | 0.953 | 400 ns | 1000 ns |
| 8 | 46.39 | 7.008 | 0.876 | 400 ns | 900 ns |

注：表中是 benchmark 的 3 次重复中位数。8 线程 uniform 的重复间波动较大（CV 11.45%，范围 38.10–47.15M），因此 7.0× 是中心估计，不是稳定保证；均值口径为 43.88M、6.70×，详见 6.5 节。

结论：

1. **静态并发读可以工作**：在关闭 `instrument`、不做并发写的前提下，BlockLearn 的 `find` 只读取构建后的结构；8 线程达到 7.01× 加速，并行效率 87.6%。
2. **这是适配器级声明，不是上游原生声明**：上游 descriptor 明确写的是 “Single-threaded”，且 `concurrent_find` 默认为 false。本实验新增的包装层显式声明只读并发，并拒绝 `instrument=1`，避免 mutable statistics 竞争。
3. **并发写未验证**：`insert/update/erase` 只在单线程路径中使用；不能把本结果外推为并发读写安全。
4. **倾斜访问的扩展性下降**：uniform 7.01×、Zipf 6.51×、hotspot 5.65×（8 线程），热点路径的缓存/内存层级竞争是主要原因。
5. **本报告不比较学习模型与 ART/HOT 的算法优劣**：ART/HOT 源码未接入，本文只测 BlockLearn 的并发只读扩展性。

---

## 2. 实验对象与版本

| 项 | 值 |
|---|---|
| Benchmark | `zsh20070718/learned-index` |
| Benchmark commit | `003a55bf674b43350fbb70242070fb74f33a56ce` |
| 被测实现 | `Zephyr761/Learned-Index-Model` |
| 模型 commit | `3894c767338a865d2920ba5e21283827fea1d803` |
| 接口 | `string-index/v1`（`src/index_contract.h`） |
| 模型 | `block-adaptive-delta-shortdir-first` |
| 适配器 | `adapter/`（`benchmark_adapter_zephyr_concurrent` 的副本） |
| 并发声明 | `capabilities.concurrent_find = true`（仅适配器层） |
| 构建 | Release，`-O3 -DNDEBUG`，GCC 16.2.0 |

### 2.1 为什么需要适配器包装

上游 BlockLearn 的 `LearnedIndex::descriptor()` 没有设置 `concurrent_find`，并写明：

```text
Single-threaded. Borrowed immutable bulk rows, stable owned insert rows.
```

其 `find` 在默认配置下只读：`instrument == false` 时不会更新 `mutable Statistics counters_`；`profile_find` 不在 benchmark 的 `find` 路径上。  
但这只能说明“当前默认路径看起来只读”，不等于上游正式承诺线程安全。因此本实验没有直接改上游声明，而是新增一个**只读并发包装层**：

- `descriptor()` 把 `concurrent_find` 设为 `true`；
- `find` 直接转发到 `LearnedIndex::find`；
- 拒绝 `instrument != 0`，防止统计计数器被并发修改；
- `insert/update/erase` 仍转发，但**不声明并发安全**；
- `bulk_load` 在启动查询线程前完成；
- 只用于 benchmark 的静态共享实例读负载。

适配器代码：

```text
adapter/
  benchmark-adapter.json
  CMakeLists.txt
  zephyr_concurrent_adapter.h
  zephyr_concurrent_adapter.cpp
```

---

## 3. 实验环境

| 项 | 值 |
|---|---|
| 机器 | Windows 11，AMD64 |
| CPU | AMD64 Family 26 Model 68（benchmark 识别为 32 个逻辑 CPU） |
| 编译器 | MinGW-w64 GCC 16.2.0（POSIX threads，UCRT） |
| CMake | 4.4.4（MinGW Makefiles 生成器） |
| Python | 3.13.12 |
| 构建类型 | Release |
| CPU 绑核 | 未绑核，继承进程亲和性 |
| 预热 | benchmark 默认无显式预热 |
| 并发方式 | 共享同一个索引实例，仅并发调用 `find` |

说明：本机 CMake + Ninja 组合在配置阶段挂起，因此使用 MinGW Makefiles；这是本机工具链问题，不是模型行为。所有配置使用同一编译器、同一构建类型和同一 runner。

---

## 4. 数据与负载

### 4.1 数据

```text
data/zephyr_keys_100k.txt
```

- 100,000 个唯一键；
- 形式：`user00000000` … `user00099999`；
- 每个键 12 字节，总键字节 1,200,000；
- 无空键、NUL、高位字节、前缀相邻对；
- 按 unsigned-byte 字典序排序、去重；
- 逻辑哈希：`e3c2121878603a5ded94dce7e9a945176f974a61da3f60f5b5fd9e984ead8e8a`。

### 4.2 负载

| 负载 | 分布 | 查询数 | 未命中 | 重复 | 延迟采样 |
|---|---|---:|---:|---:|---:|
| `uniform-200k.json` | uniform | 200,000 | 10% | 3 | 1,000 |
| `zipf-200k.json` | Zipf exponent 1.1 | 200,000 | 10% | 3 | 1,000 |
| `hotspot-200k.json` | 10% hotset / 90% 热度 | 200,000 | 10% | 3 | 1,000 |

uniform 实际为 180,000 次命中、20,000 次受控未命中，命中查询覆盖 83,485 个不同键。

每次配置都使用全新实例；并发计时从所有工作线程就绪后统一释放开始，到全部完成结束。每轮总操作数固定为 200,000，不随线程数增加。

---

## 5. 正确性验证

benchmark 在计时前执行独立校验：

- 空 build；
- 每个查询与独立参照比较；
- 未命中必须返回不存在；
- 并发查询使用同一实例，逐项比对；
- `bulk_load` 后的键值语义、scan 边界等由 benchmark 的通用校验覆盖（本实验只跑静态 `find`）。

所有自适应模型配置（uniform/Zipf/hotspot，1/2/4/8 线程）状态均为：

```text
passed
```

没有出现 wrong value、false hit、false miss 或校验异常。

---

## 6. 实验结果

### 6.1 自适应模型（主结果）

#### Uniform

| 线程数 | 吞吐量（M ops/s） | 加速比 | 并行效率 | 单线程 P50 | 单线程 P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.619 | 1.000 | 1.000 | 400 ns | 1000 ns |
| 2 | 12.404 | 1.874 | 0.937 | 400 ns | 900 ns |
| 4 | 25.245 | 3.814 | 0.953 | 400 ns | 1000 ns |
| 8 | 46.391 | 7.008 | 0.876 | 400 ns | 900 ns |

#### Zipf 1.1

| 线程数 | 吞吐量（M ops/s） | 加速比 | 并行效率 | 单线程 P50 | 单线程 P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.484 | 1.000 | 1.000 | 100 ns | 800 ns |
| 2 | 19.480 | 1.858 | 0.929 | 100 ns | 800 ns |
| 4 | 39.099 | 3.729 | 0.932 | 100 ns | 800 ns |
| 8 | 68.248 | 6.510 | 0.814 | 100 ns | 800 ns |

#### Hotspot 10% / 90%

| 线程数 | 吞吐量（M ops/s） | 加速比 | 并行效率 | 单线程 P50 | 单线程 P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.659 | 1.000 | 1.000 | 200 ns | 800 ns |
| 2 | 16.342 | 1.887 | 0.944 | 300 ns | 900 ns |
| 4 | 31.810 | 3.674 | 0.918 | 300 ns | 800 ns |
| 8 | 48.902 | 5.648 | 0.706 | 300 ns | 800 ns |

### 6.2 原规划器对照（control）

`block-residual-delta-shortdir-first`，uniform，100k：

| 线程数 | 吞吐量（M ops/s） | 加速比 | 并行效率 | P50 | P99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.893 | 1.000 | 1.000 | 300 ns | 800 ns |
| 2 | 12.661 | 1.837 | 0.918 | 300 ns | 900 ns |
| 4 | 26.132 | 3.791 | 0.948 | 300 ns | 800 ns |
| 8 | 44.767 | 6.495 | 0.812 | 300 ns | 1100 ns |

该对照用于说明：

- 自适应模型单线程约慢 4%，构建时间约 2.2 倍（170 ms vs 76–81 ms）；
- 8 线程时自适应略高（46.39M vs 44.77M），但差距在运行噪声范围内；
- 本实验**不**据此宣称自适应规划器优于原规划器；两组都只是并发实验的模型样本。

### 6.3 未包装的上游适配器

直接使用上游 `src/experiments/block-residual-delta-shortdir-first`：

| 线程数 | 状态 | 吞吐量 |
|---:|---|---:|
| 1 | passed | 6.323 M ops/s |
| 2 | unsupported | — |
| 4 | unsupported | — |

benchmark 给出的原因：

```text
adapter does not declare shared-index concurrent find support
```

这证明：**并发能力不是上游自动获得的**，而是本实验的只读包装层显式声明的。未包装版本在 2/4 线程下不会伪造性能数据。

### 6.4 构建时间与内存

自适应模型，uniform：

| 指标 | 值 |
|---|---:|
| 构建中位数 | 170.6 ms |
| `index_owned_bytes` | 676,752 B |
| `adapter_owned_bytes` | 0 B |
| `retained_input_bytes` | 5,242,880 B |

说明：

- 内存是 benchmark 的容量估算，不是 RSS；
- 5 MB 是 benchmark 持有的输入行；
- 不包含分配器头、deque 空闲块、临时训练/合并分配；
- 索引本体约 0.68 MB。

### 6.5 重复稳定性与 worker 均衡

3 次重复的波动和 worker 间 lookup 时间差异如下。`worker max/min` 是每一轮各 worker `lookup_ns` 的最大/最小值，再对轮次取平均；它不是操作数不均衡（操作数是均分的）。

| 负载 | 线程 | 中位数（M） | 均值（M） | CV | 范围（M） | worker max/min |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1 | 6.619 | 6.552 | 2.10% | 6.394–6.642 | 1.000 |
| uniform | 2 | 12.404 | 12.477 | 2.98% | 12.146–12.880 | 1.016 |
| uniform | 4 | 25.245 | 25.023 | 4.61% | 23.776–26.049 | 1.065 |
| uniform | 8 | 46.391 | 43.879 | **11.45%** | **38.095–47.151** | 1.126 |
| Zipf | 1 | 10.484 | 10.501 | 0.29% | 10.483–10.536 | 1.000 |
| Zipf | 2 | 19.480 | 18.435 | 13.34% | 15.627–20.198 | 1.038 |
| Zipf | 4 | 39.099 | 37.861 | 9.57% | 33.782–40.702 | 1.106 |
| Zipf | 8 | 68.248 | 67.641 | 2.70% | 65.591–69.085 | 1.159 |
| hotspot | 1 | 8.659 | 8.657 | 0.32% | 8.629–8.684 | 1.000 |
| hotspot | 2 | 16.342 | 16.131 | 3.16% | 15.549–16.502 | 1.077 |
| hotspot | 4 | 31.810 | 30.644 | 7.09% | 28.138–31.984 | 1.094 |
| hotspot | 8 | 48.902 | 48.738 | 5.37% | 46.042–51.272 | 1.262 |

解释：

- uniform 的 1/2/4 线程 CV 为 2–5%，比较稳定；8 线程 CV 升到 11.45%，主要是共享机器调度、频率和内存通道状态造成，不能忽略。
- Zipf 的 2/4 线程 CV 较大（13.34%/9.57%），但中位数与均值差距明显；8 线程反而更稳定（2.70%）。
- hotspot 的 8 线程 worker max/min 为 1.262，高于 uniform 的 1.126，说明热点路径上不同 worker 的实际执行时间差异更大；这与“热点数据/缓存行更集中”一致。
- 3 次重复不足以给出严格置信区间；本报告以 benchmark 的中位数为准，并同时给出均值、CV 和范围，避免把单次结果当成稳定结论。

---

## 7. 分析

### 7.1 Uniform：接近线性，8 线程开始受内存层级限制

- 2 线程 1.87×、4 线程 3.81×，4 线程并行效率 95.3%；
- 8 线程 7.01×，效率 87.6%；
- 吞吐量从 6.62M 提升到 46.39M ops/s。

4 线程出现略高于线性的 3.81×，主要来自测量噪声和频率/缓存状态差异，不应解释为超线性算法性质。8 线程仍保持 87.6% 效率，说明在 100k 键、12 字节键的规模下，共享只读路径没有出现严重的锁或全局计数器竞争。

### 7.2 Zipf：单线程更快，8 线程效率下降

- 单线程 10.48M，高于 uniform 的 6.62M，原因是 Zipf 命中集中在少量热键，缓存局部性更好；
- 8 线程 68.25M，6.51×，效率 81.4%；
- 线程增加后热点键的缓存行/内存通道竞争更明显，效率低于 uniform。

### 7.3 Hotspot：8 线程效率最低

- 单线程 8.66M；
- 8 线程 48.90M，5.65×，效率 70.6%；
- 90% 访问集中在 10% 键，工作线程反复命中相同叶子和 tag 数组，缓存一致性/内存带宽成为主要瓶颈；
- 但 P50 仍为 200–300 ns，P99 800–900 ns，没有出现数量级退化。

### 7.4 延迟口径

表中的 P50/P99 来自 benchmark 的**独立单线程抽样实例**，不是 8 线程并发负载下的尾延迟。  
因此不能把表中的 P99 解释为“8 线程下的 P99”；它用于说明查询本身的延迟量级。

### 7.5 与并发写的关系

本实验只覆盖共享实例的静态只读：

- `bulk_load` 在查询线程启动前完成；
- 查询阶段没有 `insert/update/erase`；
- 没有并发写、没有写入后的并发读、没有重建；
- 所以本结果与“并发读写安全”无关。

---

### 7.6 边际扩展与每线程吞吐

用中位数计算每次线程数翻倍的边际加速：

| 负载 | 1→2 | 2→4 | 4→8 | 4→8 边际效率 |
|---|---:|---:|---:|---:|
| uniform | 1.874× | 2.036× | 1.838× | 91.9% |
| Zipf | 1.858× | 2.007× | 1.745× | 87.3% |
| hotspot | 1.887× | 1.947× | 1.538× | 76.9% |

每线程吞吐（M ops/s/线程）：

| 负载 | 1 线程 | 2 线程 | 4 线程 | 8 线程 |
|---|---:|---:|---:|---:|
| uniform | 6.62 | 6.20 | 6.31 | 5.80 |
| Zipf | 10.48 | 9.74 | 9.78 | 8.53 |
| hotspot | 8.66 | 8.17 | 7.95 | 6.11 |

观察：

- uniform 从 1 到 4 线程的每线程吞吐基本持平（6.2–6.3M），说明没有明显的全局锁竞争；到 8 线程降到 5.80M（−12%），更像内存/缓存和调度成本。
- hotspot 的每线程吞吐从 8.66M 降到 6.11M（−29%），热点访问的缓存一致性/带宽压力最明显。
- Zipf 的每线程吞吐从 10.48M 降到 8.53M（−19%），仍高于 uniform 的绝对吞吐，因为热键局部性降低了单次查询成本。

4 线程 uniform 的 2.036× 边际加速略高于 2×，属于 3 次重复下的测量波动，不应解释为超线性性质；同一配置的 CV 为 4.61%，足以解释这个偏差。

### 7.7 自适应模型与原规划器对照的进一步解读

| 指标 | 原规划器（control） | 自适应 | 差异 |
|---|---:|---:|---:|
| 单线程吞吐 | 6.893M | 6.619M | −4.0% |
| 8 线程吞吐（中位数） | 44.767M | 46.391M | +3.6% |
| 8 线程加速比 | 6.495× | 7.008× | +0.51× |
| 8 线程效率 | 81.2% | 87.6% | +6.4 pp |
| 构建时间 | 76–81 ms | 170.6 ms | 约 2.2× |
| 索引内存 | 675,792 B | 676,752 B | +960 B |
| 单线程 P50 / P99 | 300 / 800–1100 ns | 400 / 900–1000 ns | P50 +100 ns |

结论应谨慎：

- 自适应模型单线程更慢、构建更贵，这是它的规划/标签成本；
- 8 线程中位数略高，但自适应 8 线程 CV 为 11.45%，对照为 5.84%，**3.6% 的差异不足以证明自适应模型在并发上更优**；
- 两者在 8 线程的绝对差距（约 1.6M ops/s）小于重复间波动；
- 本实验的设计目标是测并发能力，不是做规划器消融；不能根据这组数据给 P0/P2 排名。

### 7.8 对“并发能力”的解释

这组数据支持的是：

> 在静态只读、`instrument=false`、构建完成后不再写入的条件下，BlockLearn 的 `find` 可以在共享实例上并发扩展；uniform 下 8 线程中位数约 7×，但高线程数有明显方差。

它不支持：

- 并发写安全；
- 写后读/重建期间的并发安全；
- 任意调度下无数据竞争；
- 在 Linux/其他 CPU 上同样的绝对吞吐或扩展比；
- 在真实 URL/词典数据上同样的 Zipf/hotspot 表现。

---

## 8. 结论

1. **最新版 BlockLearn 的默认 `find` 路径可以作为共享实例的并发只读查询路径**：在 `instrument=false`、无并发写、`bulk_load` 完成后再启动线程的条件下，100k 键、200k 查询的 8 线程 uniform 吞吐中位数为 46.39M ops/s，7.01× 加速；但 8 线程 CV 为 11.45%，均值口径为 43.88M / 6.70×。
2. **扩展性在 4 线程最好（95.3% 效率），8 线程仍可接受（87.6%）**；Zipf/hotspot 在 8 线程分别降到 81.4%/70.6%，说明热点访问下缓存和内存层级是主要限制。
3. **不能把它称为上游原生并发能力**：上游明确声明 single-threaded，`concurrent_find` 默认为 false；本实验通过包装层显式打开只读并发，并禁止 `instrument` 以规避 mutable statistics。
4. **写路径没有并发结论**：`insert/update/erase` 未在并发下验证；如果后续要支持并发写，需要单独设计同步/版本协议并做线性化或 TSAN 验证。
5. **与 ART/HOT 的算法对比不在本报告范围内**：本机未接入 ART/HOT 源码，报告只回答“该学习索引在 benchmark 下的并发只读能力”。

---

## 9. 限制与风险

1. **适配器级并发声明**：`concurrent_find` 是包装层声明的，不是上游正式声明；上游 README 也明确“并发不在本轮实现范围”。
2. **只验证了只读**：没有并发写、并发 scan、写后读、重建与并发查询的组合。
3. **没有 ThreadSanitizer/ASan**：benchmark 的独立校验和压力运行只能说明本次负载下结果正确，不构成任意调度的数据竞争证明。
4. **只有 3 次重复**：8 线程 uniform 的 CV 为 11.45%，范围 38.1–47.2M；报告的 7.0× 是中位数中心估计，不能当作稳定 SLA。Zipf 的 2/4 线程 CV 也达到 13.3%/9.6%。
5. **平台限制**：Windows + MinGW-w64 GCC 16；未在 Linux x86-64 上重复；CPU 未绑核，机器有其他负载。
6. **数据规模有限**：100k 键、12 字节、合成 `user%08d`；真实 URL/词典/短语和更大规模可能改变缓存行为。
7. **没有 ART/HOT 对照**：本机未准备 ART/HOT 源码；报告不比较算法，只比较 BlockLearn 自身的线程扩展。
8. **延迟口径**：P50/P99 是独立单线程采样，不是并发尾延迟。
9. **内存口径**：benchmark 容量估算，不是 RSS 或峰值内存。
10. **自适应模型依赖本机校准文件**：`planner-costs.txt` 在本机生成；跨机器不能直接复用。

---

## 10. 复现

下列命令以原始工作区路径为例；上传包中的适配器源码在 `adapter/`，运行时把 `--implementation` 指向 `adapter/` 所在目录即可。自适应模型需要独立校准文件。

数据集可用 [generate_dataset.py](generate_dataset.py) 重新生成；负载 JSON 在 `workloads/`。

### 10.1 准备

```powershell
# 本实验使用的 POSIX MinGW 工具链
$win = "...\_deps\winlibs\mingw64\bin"
$env:PATH = "$win;$env:PATH"
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CC = "$win\gcc.exe"
$env:CXX = "$win\g++.exe"

# 自适应模型需要独立校准文件
$env:BLOCKLEARN_COST_PROFILE = "...\zephyr_workloads\planner-costs.txt"
```

### 10.2 运行 benchmark

```powershell
python -B learned-index\scripts\benchmark.py `
  --dataset data\zephyr_keys_100k.txt `
  --implementation adapter `
  --workload zephyr_workloads\uniform-200k.json `
  --threads 1 2 4 8 `
  --jobs 2 `
  --build-timeout 900 `
  --run-timeout 600 `
  --output results\zephyr-adaptive-100k-uniform
```

Zipf / hotspot 只需替换 `--workload` 和 `--output`。

### 10.3 未包装对照

```powershell
python -B learned-index\scripts\benchmark.py `
  --dataset data\zephyr_keys_100k.txt `
  --implementation ...\Learned-Index-Model-main\src\experiments\block-residual-delta-shortdir-first `
  --workload zephyr_workloads\uniform-200k.json `
  --threads 1 2 4 `
  --output results\zephyr-raw-concurrency
```

### 10.4 生成校准文件

```powershell
...\zephyr-gcc16\src\blocklearn_calibrate.exe `
  --output zephyr_workloads\planner-costs.txt
```

---

## 11. 结果文件

- [uniform 自适应](raw/zephyr-adaptive-100k-uniform/report.md)
- [Zipf 自适应](raw/zephyr-adaptive-100k-zipf/report.md)
- [hotspot 自适应](raw/zephyr-adaptive-100k-hotspot/report.md)
- [uniform 原规划器对照](raw/zephyr-100k-uniform-8/report.md)
- [未包装上游适配器](raw/zephyr-raw-concurrency/report.md)
- [汇总 CSV](summary.csv)
- [adaptor manifest](adapter/benchmark-adapter.json)
- [adaptor 源码](adapter/zephyr_concurrent_adapter.cpp)

原始 CSV/JSON：

```text
raw/zephyr-adaptive-100k-uniform/{report.md,summary.csv,scaling.csv,workers.csv}
raw/zephyr-adaptive-100k-zipf/{report.md,summary.csv,scaling.csv,workers.csv}
raw/zephyr-adaptive-100k-hotspot/{report.md,summary.csv,scaling.csv,workers.csv}
raw/zephyr-100k-uniform-8/{report.md,summary.csv,scaling.csv,workers.csv}
raw/zephyr-raw-concurrency/{report.md,summary.csv,scaling.csv}
```
