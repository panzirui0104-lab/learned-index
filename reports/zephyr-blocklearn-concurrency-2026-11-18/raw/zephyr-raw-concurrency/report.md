# 字符串索引统一评测报告

本报告覆盖内存、唯一键 map 的静态点查询，支持单线程和共享索引实例的并发只读查询。验收 `bulk_load` 和 `find`。

## 数据与负载

- Reader：raw-lines
- 原始记录：100000；排序去重后键数：100000；重复记录：0
- 逻辑哈希（SHA-256）：`e3c2121878603a5ded94dce7e9a945176f974a61da3f60f5b5fd9e984ead8e8a`
- 负载：`static-point/v2`；请求数：200000；受控未命中：20000；重复：3
- 访问分布：均匀访问（uniform）
- 实际命中：180000；实际未命中：20000；命中访问覆盖 83485 个不同键。
- 查询序列哈希（SHA-256）：`20ecae2b3ac9ee70014628224210a35f551471faf016c4f44be5b7175ace82e8`
- 查询线程数：1, 2, 4；各配置依次运行，每个适配器仅编译一次。
- 每轮总请求数固定，连续均分给工作线程，不随线程数增加。

## 实现结果

| 实现 ID | 线程数 | 状态 | 阶段 | 原因 | 构建中位数（ns） | 吞吐量中位数（ops/s） | 吞吐量范围（ops/s） | 吞吐量标准差 | 独立单线程 P50（ns） | 独立单线程 P99（ns） |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block-residual-delta-shortdir-first | 1 | passed | complete | 不可用 | 81729700 | 6322611.238441477 | 6299212.5984251965–6459134.669728297 | 70522.81942338671 | 300 | 1800 |
| block-residual-delta-shortdir-first | 2 | unsupported | capability | adapter does not declare shared-index concurrent find support | 不可用 | 不可用 | 不可用–不可用 | 不可用 | 不可用 | 不可用 |
| block-residual-delta-shortdir-first | 4 | unsupported | capability | adapter does not declare shared-index concurrent find support | 不可用 | 不可用 | 不可用–不可用 | 不可用 | 不可用 | 不可用 |

## 线程扩展性

加速比 = 当前吞吐量中位数 / 同一实现的单线程吞吐量中位数；并行效率 = 加速比 / 线程数。未请求单线程、单线程失败或总操作数不同时留空。

| 实现 ID | 线程数 | 状态 | 加速比 | 并行效率 |
| --- | ---: | --- | ---: | ---: |
| block-residual-delta-shortdir-first | 1 | passed | 1.0 | 1.0 |
| block-residual-delta-shortdir-first | 2 | unsupported | 不可用 | 不可用 |
| block-residual-delta-shortdir-first | 4 | unsupported | 不可用 | 不可用 |

## 解读限制

失败、跳过和不支持的实现保留在所有汇总文件中，性能列为空，不以零值代替。内存分项只有在适配器给出明确口径时才展示；本报告不据此生成内存排名。进程未绑核，小规模结果不用于论文级排名。
并行吞吐量计时从所有工作线程就绪后的统一释放开始，到全部完成为止；不含创建/回收线程及构建，包含调度和同步开销。延迟来自另建索引的独立单线程抽样，不是并发负载下的尾延迟；延迟包含计时器开销，默认不预热。并发校验通过只说明本次工作负载通过，不证明实现没有数据竞争。
`workers.csv` 保留每轮各线程的操作数、耗时与校验和；`scaling.csv` 可直接绘制扩展性曲线。运行环境和继承的 CPU 亲和性见 `report.json`。
