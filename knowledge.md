# 多模态向量索引压缩方案知识库

日期：2026-10-14

## 1. 先明确压缩对象

本题压缩的是**整个可独立检索的索引目录**。压缩包小、常驻内存小、把向量放到 SSD、把大文件移出 index/，都不能代替实际索引字节数合规。检索不能读取官方原始 base；因此用于重排的向量表示也必须计入索引。训练只使用公开 base，公开 query/GT 只用于评测、参数选择。

Final-10M 的 FP32 裸向量为 10,000,000 × 1024 × 4 = 40.96GB（十进制）。600MB 对应整个索引平均 60B/条，需约 68.27 倍压缩；2000MB 对应 200B/条，需约 20.48 倍。ID、粗聚类中心、PCA/旋转矩阵、量化码本、图边、辅助系数都占空间。这里 MB=10^6B；最终仍应以官方校验器计量为准。

附件的上限存在 1000M、800MB、2000MB 三种说法。本文以最后评分表及已有分析的 2000MB 为工作假设，600MB 满分目标不变；不能将此假设当成官方澄清。

## 2. 量化方法与预算

| 表示 | 向量编码字节/条（1024维） | Final-10M 编码体积 | 主要用途/限制 |
|---|---:|---:|---|
| FP32 | 4096 | 40.96GB | 精确检索参考，无法提交 |
| FP16 | 2048 | 20.48GB | 精度通常较好，空间仍超限 |
| SQ8 | 1024 | 10.24GB | 每维8bit，是精度基线 |
| SQ4 | 512 | 5.12GB | 每维4bit，仍超2000MB |
| 1bit RaBitQ | 128 + 辅助系数 | 至少1.28GB | 原始1024维有机会满足准入，无法仅靠1bit达600MB |
| PQ32×8 | 32 | 320MB | 32个子空间、每个1字节，压缩很强 |
| PQ64×8 | 64 | 640MB | 编码本身已超600MB |
| PQ128×8 | 128 | 1.28GB | 较高精度的准入路线 |
| PCA256+SQ4 | 128 | 1.28GB | 降维和量化会同时损失精度 |

Faiss IVF 通常再存8B ID/条；例如 PQ32 对应约40B/条，PQ128约136B/条，另加固定开销。上述 RaBitQ 数字只是 bit 编码的下界，具体库、位数、块填充与辅助量不同。公式与结构可核对 [Faiss 索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。

### SQ：标量量化

按每维的训练分布映射到有限整数档位，搜索时用量化表示近似距离。SQ8/SQ4实现简单，可建立“精度和耗时尚可但空间不够”的参照。均匀编码、截断范围和离群值处理会影响误差，不能假设SQ8必达90%。降低维度后再SQ能缩小体积，但必须测近邻排名，而不是只看重建误差。

### PQ：乘积量化

把维度划成 M 个子向量，每个子空间学习 2^b 个中心，存中心编号。编码约 M×b/8B；查询生成各子空间距离表，累加得到近似距离（ADC）。增加 M 往往降低误差但增大体积；增大 nprobe 只能减少漏搜，无法还原被量化丢掉的信息。训练样本量和迭代次数也是关键参数。[Faiss 量化构件](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks%3A-clustering%2C-PCA%2C-quantization)

### PCA / OPQ / 分层重排

PCA保留高方差主成分，但总体解释方差不是 Recall@10。相近向量之间的距离差可能恰好在被丢弃的低方差方向。OPQ学习旋转以改善PQ子空间划分，保持全维时不直接压缩空间；与降维组合则需要额外精度验证。不能默认 `PCA256` 或 `OPQ` 一定有效。[Faiss index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

可先用短码召回候选，再用更高精度的**压缩表示**重排，或使用 IVFPQR。短码+重排码的每条总字节必须一起计入；保留1024维FP16/FP32做重排在本题空间预算下行不通。

### RaBitQ

通过旋转、相对中心的量化和校正系数估计距离；提供理论误差分析，但“误差有界”不等于任意数据与任意bit数必达90%召回。可以配合IVF、图索引、多bit编码。1024维1bit编码已经128B/条，满分路线需要降维、非均匀位分配或其他结构，不能直接宣称只需60B。

官方论文：[RaBitQ](https://arxiv.org/abs/2405.12497)。官方实现：[RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library)，[IVF使用说明](https://vectordb-ntu.github.io/RaBitQ-Library/index/ivf/)。在线项目页面声称支持ARM64 NEON，但必须针对比赛实际版本和CPU验证。本机已装 Faiss 1.10.0，没有 `IndexRaBitQ`；本轮没有将其他算法冒充RaBitQ，也没有为追最新版修改全局环境。

## 3. 候选导航结构

### IVF

先聚类分桶，查询只扫描 nprobe 个桶。nlist↑通常减少单桶扫描量，也增加训练和路由开销。nprobe↑提升桶覆盖率，但拉长延迟；nprobe=nlist 时扫描全部桶，可用于观察压缩表示的召回极限。IVF和PQ/SQ/RaBitQ是可组合的不同组件。

### HNSW / 压缩图

图导航可以减少扫描量，但图边的预算很大：每条 R 个32bit邻居至少4R B/条。例如R=16仅邻居ID就64B，10M约640MB，尚未包括量化码、层级和索引头。常规HNSW+SQ8无法满足本题。应先核算实际序列化布局，再评估小度数图、边压缩或更短编码的取舍。[HNSW论文](https://arxiv.org/abs/1603.09320)

### DiskANN

SSD图搜索值得借鉴，但低内存占用并不意味着低磁盘体积。如果配置在磁盘保存全精度向量用于距离或重排，仍会超本题预算。必须计入图、码本、磁盘向量、缓存持久化文件。当前官方仓库与题目时代的C++实现已经存在分支差异，应固定版本和文件布局。[DiskANN官方仓库](https://github.com/microsoft/DiskANN)，[2019论文](https://arxiv.org/abs/1907.09390)

### AlayaLaser / AlayaLite

可借鉴高维距离计算、压缩图布局、SSD访问与缓存策略。题目给出的“约100B/条”是目标性估算，不是本数据已验证成绩；100B×10M也约1000MB，不是满分600MB。在线AlayaLite的LASER平台列表没有保证比赛鲲鹏ARM可直接安装，移植成本必须实测。论文/项目性能不能照搬成本题 batch=1 的p95。[AlayaLaser论文](https://arxiv.org/abs/2602.23342)，[AlayaLite仓库](https://github.com/AlayaDB-AI/AlayaLite)，[LASER指南](https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/LASER.md)

## 4. 本题优先研发顺序

1. 原始维度IVF+SQ8建立召回参照，先证明数据、距离、ID顺序正确。
2. 原始维度IVF+PQ64/PQ128试准入预算；更短PQ试满分预算。先看全桶扫描能否达90%，失败则优先改善编码。
3. 对有潜力的编码再增加OPQ、残差/分层量化和压缩码重排；用固定验证查询选择参数，留出部分公开query检查泛化。
4. 再试固定版本RaBitQ、低度数压缩图、AlayaLaser；先ARM编译和保存/加载小样，再投入10M构建。
5. 正式数据单独训练、构建、评测。Dev-1M并非Final-10M的保证嵌套子集，不能复用其GT评价Final。

最终比较应同时报告完整文件字节、Recall@10、batch=1/concurrency=1 p50/p95、构建耗时、加载耗时与增量RSS。通过二进制stdin/stdout协议的端到端延迟需要单独验证。所有依赖和程序必须在鲲鹏openEuler离线环境中可用；Faiss提供Linux aarch64支持，但不等于任意openEuler版本都能直接安装现成包。[Faiss安装文档](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
