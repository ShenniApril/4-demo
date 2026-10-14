"""Generate the report tables directly from recorded full-query measurements."""
import json
from pathlib import Path

out=Path('experiments')
rows=[]
for p in out.glob('*_np*.json'):
    if p.stem.endswith('_repeat'): continue
    r=json.loads(p.read_text()); r['result_file']=p.name; rows.append(r)
rows.sort(key=lambda r:(r['algorithm'],r['nprobe']))
assert len(rows)==21, 'All seven configurations and three probes must finish before reporting.'
assert all(r['query_count']==1000 and len(r['latencies_ms'])==1000 for r in rows)
codes={'PCA256,IVF256,PQ32':32,'PCA256,IVF256,PQ64':64,
       'PCA256,IVF256,SQ4':128,'PCA256,IVF256,SQ8':256,
       'IVF256,PQ64':64,'IVF256,PQ128':128,'IVF256,SQ8':1024}

def projection(r): return (r['index_bytes']+9000000*(codes[r['algorithm']]+8))/1e6

table='| 方案 | nprobe | 完整索引 MB | B/条 | Recall@10 | p50 ms | p95 ms | 10M空间外推 MB |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in rows:
    table+=f"| {r['algorithm']} | {r['nprobe']} | {r['index_bytes']/1e6:.3f} | {r['bytes_per_vector']:.3f} | {100*r['recall_at_10']:.2f}% | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {projection(r):.3f} |\n"
build='| 方案 | 量化/IVF训练 s | 构建 s | 首次加载 s |\n|---|---:|---:|---:|\n'
for r in rows:
    if r['nprobe']==16:
        build+=f"| {r['algorithm']} | {r['train_sec']:.2f} | {r['build_sec']:.2f} | {r['load_sec']:.3f} |\n"
independent=json.loads((out/'independent_load.json').read_text()) if (out/'independent_load.json').exists() else {}
assert independent.get('base_opened') is False, 'Independent verification must finish.'
thread1=json.loads((out/'threads_1.json').read_text())
thread4=json.loads((out/'threads_4.json').read_text())
sq=next(r for r in rows if r['algorithm']=='IVF256,SQ8' and r['nprobe']==16)
feasible=[r for r in rows if r['recall_at_10']>=.9 and r['p95_ms']<=600 and projection(r)<=2000]
assert not feasible, 'Update conclusion if a feasible configuration is found.'
text=f'''# Dev-1M 赛前可行性评估与实测报告

评估日期：2026-10-14。使用本目录官方开发库，完整1,000,000条base、1,000条公开query。题目截图与已有分析作为约束和背景材料，在线项目介绍作为方案参考；所有实测数字由本地程序产生。

## 1. 结论

本地开发有价值，可以完成数据校验、压缩效果比较与算法筛选；最终性能和运行兼容性必须在官方鲲鹏ARM/openEuler服务器确认。没有访问官方服务器，不能承诺它比本机快几倍或慢多少百分比。

本轮覆盖7种配置、每种3个nprobe，所有召回和延迟测量使用全部1000条公开query。**没有发现同时满足90%召回与10M外推≤2000MB的配置**。原始维度IVF+SQ8在nprobe=16达到{sq['recall_at_10']*100:.2f}%召回、本地p95={sq['p95_ms']:.2f}ms，但10M空间外推约{projection(sq):.0f}MB，严重超限；PCA256+PQ32空间外推约406MB，却只有59.43%全桶召回。精度和空间必须由同一个配置同时通过。RaBitQ/AlayaLaser/DiskANN本轮完成资料与预算评估，未在此数据上跑出成绩。

## 2. 官方约束核对

| 项目 | 当前理解 |
|---|---|
| 正式规模 | 10,000,000×1024 FP32，Squared L2，返回10个合法不重复ID |
| 精度 | 严格 Recall@10≥90%，与GT前10个ID做交集后除以10，再取查询平均 |
| 延迟 | batch=1、concurrency=1、预热后p95≤600ms |
| 空间 | 600MB满分；最终评分段2000MB准入；整个解压后的index/计量 |
| 运行环境 | 鲲鹏ARM，openEuler22.03 64bit，80G SSD；具体型号、核心数、RAM未提供 |
| 提交 | 已构建完整索引、config、manifest、程序和离线依赖；评委不重建不调参 |
| 搜索 | stdin/stdout二进制官方协议；不读取原始base、隐藏GT、外网 |

**待官方澄清**：截图4-2写1000M，4-3写800MB，4-5/4-8和已有4.md写2000MB；本文按最后评分段2000MB做可行性判断。这不是官方确认。4-5流程还写2000M准入检查。若最终采用800MB或1000MB，准入方案选择将明显变化。

## 3. 环境与实验方法

- 本机：Intel Core Ultra 5 125H，18逻辑CPU，Windows11 x86-64；物理内存约31.46GiB。初次观察可用约10.57GiB，随其他应用占用波动；D盘空闲约133.86GB。
- Python3.12.7、NumPy1.26.4、Faiss1.10.0（AVX2）、MKL2023.1。版本记录见 `experiments/environment.json` 和 `requirements-benchmark.txt`。
- 训练从base固定seed=42均匀无放回采样32,768条；IVF256，k-means/PQ均12次迭代，训练与添加限制4线程。PCA256也只用这批base训练。公开query不参与PCA/PQ/IVF训练。
- 实际添加全部100万条base；每个配置保存完整Faiss索引后重新加载。评价时仅调用加载后的索引，不拿原始base重排。
- 搜索限制Faiss与BLAS为1线程，单条逐次调用，预热20条，再按seed=2026打乱的顺序评价全部1000条。每个配置测一轮。记录全部单条延迟、每条召回及返回ID到JSON。
- 第一组PCA256+SQ8曾与约6秒的数据精确校验交叠，完成全部构建后在独立进程重新测量其三档nprobe，表中使用干净复测延迟；两次召回一致，复测JSON也保留。
- 记录索引文件的实际字节和SHA-256。表中空间包括文件内变换矩阵、中心、ID与码本；外部 `pca256.transform` 是训练中间产物，不是已保存索引的运行依赖。
- 延迟是Python内直接调用索引的时间，包含查询变换，但不包含官方搜索进程二进制协议传输。它是算法基准，不能称为官方端到端成绩。

三份数据文件实际字节和SHA-256全部匹配官方manifest（`experiments/data_checks.json`）。块式FP32全量精确Squared L2验证前100条公开query，Recall@10=1.0，耗时约6.01s。该步骤用于数据/GT冒烟验证，不是允许提交的检索方案；不是逐查询延迟，也未执行完整1000条FP64审计。

本机同时载入Intel与LLVM OpenMP，threadpoolctl发出兼容性警告；本轮各运行均完成，未遇到崩溃。它仍是复现环境限制，官方环境应避免混装并固定运行时。正式依赖只读安装文档并不能代替openEuler实装验证。

笔记本后台负载和温度/频率未严格隔离。PCA256+SQ8复测的部分p95比首次更高，即使没有其他实验同时运行也存在波动；这些单轮延迟用于可行性筛选，不是稳定排名。正式服务器应固定线程与CPU亲和性并按官方规则复测。

## 4. 完整结果

MB使用十进制。nprobe=256表示扫描全部256个桶，能用于观察当前压缩表示的精度极限；不是精确FP32检索。

{table}

10M空间外推模型：`S10 = S1 + 9,000,000×(编码字节+8字节ID)`，维度、nlist、码本和文件格式保持不变。它避免把PCA矩阵等固定开销重复乘10。真实Final需增加聚类数或改变结构时，空间会相应变化；实际文件字节才是最终依据。这里没有构建10M，也没有测其Recall或延迟。

{build}

PCA单独训练约4.51s，不计入表中各配置的构建耗时；构建列包含IVF/PQ/SQ训练及全部添加，不包含序列化、数据SHA校验、评测、独立PCA训练。加载时间是缓存可能已热的观测值，不是冷启动SSD能力。主实验RSS包含内存映射base及训练工作区，不能当作提交检索增量RSS。

本轮仅256个粗中心、32,768条训练样本和12次迭代，因此构建明显快于已有4.md里的宽泛时间估算。它不代表高质量码本、更多粗中心或Final10M也能在相同时间完成；尤其PQ召回较低，应该增加训练强度再比较，而不是把本轮默认配置视作算法上限。

PCA训练样本解释方差：256维约76.17%、512维90.93%、768维97.18%。PCA256+SQ8即使全桶扫描Recall也只有78.89%，显示当前配置的表示精度不足，不能靠增加nprobe解决。PQ32虽很小，但全桶Recall59.43%；压缩达到满分体积不代表精度可准入。

## 5. 独立加载验证

`evaluate_saved.py` 在独立进程只读已保存索引、公开query和开发评测GT，代码不打开base.fbin。GT仅用于客户端计分，不进入索引搜索。验证的是索引不依赖原始base，不是已实现官方提交搜索服务。

独立验证使用原始1024维IVF256+PQ128、nprobe=64、1线程、全部1000条query：Recall@10={independent['recall_at_10']*100:.2f}%，p50={independent['p50_ms']:.2f}ms，p95={independent['p95_ms']:.2f}ms；加载约{independent['load_sec']:.3f}s，加载RSS增量约{independent['load_rss_delta_bytes']/1e6:.1f}MB。完整记录见 `experiments/independent_load.json`。

PCA256+SQ8的独立复测也成功完成；程序只加载该Faiss索引，未读取单独的PCA训练文件，说明变换已包含在索引中。

加载RSS增量仅指本次进程加载前后的差值，不是峰值RSS，也不包含整机缓存。目录中没有官方tools/和search.sh模板，因此本轮未执行官方接口/manifest校验器，没有制作Final提交包。

## 6. 换到华为服务器会造成多大差异

### 不同指标的可迁移程度

| 指标 | 从本机迁移到鲲鹏的影响 |
|---|---|
| 已保存相同文件的索引体积 | 文件复制后的字节与SHA不变；重新训练或改变版本需重测 |
| Recall | 同一索引与相同查询理论上接近；浮点实现、近似路由和距离并列可能改少量ID，需比对 |
| 构建时间 | CPU、内存带宽、BLAS、线程与热降频均影响，不能按核心数比例换算 |
| 单条p95 | AVX2与ARM NEON内核、主频、缓存、NUMA、线程和后台负载都影响，最不能直接复制 |
| 安装/启动 | Windows DLL/exe不能在ARM Linux运行；需ARM包或现场编译，shell与离线依赖也需适配 |

Faiss官方提供Linux aarch64支持，属于可移植路线，但本机的Windows AVX2构建不是可上传的运行二进制。[Faiss安装资料](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

多核服务器未必使batch=1/concurrency=1搜索按核心数加速。训练/批量添加可能受益更多；单条检索的并行策略依索引和实现而异，应比较1/2/4/自动线程的实际p95。[Faiss线程说明](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls)

本机同一PQ128索引、nprobe64、相同200条查询的一轮对照：1线程p95={thread1['p95_ms']:.2f}ms，4线程p95={thread4['p95_ms']:.2f}ms，Recall分别为{thread1['recall_at_10']*100:.2f}%和{thread4['recall_at_10']*100:.2f}%。这是线程设置敏感性检查，不是跨CPU实验，样本/轮次也不足以形成稳定的加速比结论。

### 延迟敏感性：明确是模型，不是服务器实测

IVF平均扫描候选约 `N×nprobe/nlist`。如果10M仍保持256桶和相同nprobe，扫描量大致变成10倍。定义α为鲲鹏相对本机处理同等扫描量的耗时倍率；忽略固定开销、桶不均匀、缓存变化时，`T10≈10×α×T1`。α现在未知。

例如本地1M p95为20ms，则该简化模型下，α=0.5/1/2时10M约100/200/400ms；本地1M为60ms则约300/600/1200ms。这是说明如何做余量评估的假设算例，不是本轮某配置成绩，更不是鲲鹏速度结论。

提高nlist可控制扫描量，但需要新训练、更大中心/路由开销，并可能需要提高nprobe维持召回。因此1M到10M既不能直接等延迟，也不能保证严格10倍。应该把“CPU平台差异”和“数据库规模差异”分开做对照。

80G SSD对40.96GB原始Final数据并不宽裕：若同时保留一个接近原始体积的归档和解压base，约81.92GB已经超过80G，尚未包括系统、索引与临时文件。采用流式下载/解包、分片顺序处理，规划可用空间；不应假设80G硬盘意味着80G空闲。RAM规格未知，无法给出全量构建是否能一次放入内存的确定答案。

建议以同一个小样索引首先验证ARM安装、序列化兼容和协议，再在同一Dev数据、相同参数/查询顺序上比较本机与服务器Recall/p95；最后用Final10M单独构建和公开Final query/GT（若提供）评价。构建可在本机或其他获授权设备完成，只要最终程序能在规定环境独立运行且索引匹配Final版本；截图明确要求在评测同CPU/OS编译，故运行库应在ARM/openEuler构建验证。

## 7. 赛前决策与后续验证

本轮能证明本机可快速完成1M实验闭环、普通高压缩配置存在明显召回损失；不能证明600MB满分方案已可用，也不能证明整个赛题不可行。训练样本/迭代量、OPQ、更优码本、残差/分层量化、多bit RaBitQ、压缩图都有改进空间。

优先依据全桶召回筛选编码：全桶仍低于90%的当前配置，应先改编码再调导航。精度有余量但超空间的配置，可作为上界参照。后续设独立公开query验证分组，避免在所有公开query反复选参后误判泛化。正式数据分布与Dev不同，必须重新训练和测试，不能用Dev GT评价Final。

上线前必须完成：官方上限澄清；获得CPU/RAM/可用SSD参数；离线ARM依赖与编译验证；完整Final构建；原始base不可见时的官方协议端到端检索；全部索引文件的官方manifest/SHA检查。评委仅加载索引，不能现场补建或补文件。

复现（在本目录执行，耗时随环境变化）：

```powershell
python audit.py
python benchmark.py --mode initial
python benchmark.py --mode pq
python run_remaining.py
python final_checks.py
python make_report.py
```

主要产物：`knowledge.md`、本报告、`experiments/*.faiss` 完整开发索引、`experiments/*_np*.json`逐查询原始结果、`experiments/independent_load.json`独立加载验证。所有实验脚本均保留。不要把整个experiments目录作为提交index/，其中包括训练中间文件和评测GT统计。
'''
Path('report.md').write_text(text,encoding='utf-8')
summary=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('report.md generated with',len(rows),'runs')
