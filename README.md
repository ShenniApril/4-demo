# Dev-1M 向量索引压缩实验

2026-10-14 在官方 Dev-1M 数据上完成的赛前可行性评估。

## 内容

- [knowledge.md](knowledge.md)：SQ/PQ/PCA/OPQ/RaBitQ、图索引和 SSD 路线的原理、预算与资料。
- [report.md](report.md)：服务器迁移影响、完整测量表、局限及后续建议。
- [experiments/summary.json](experiments/summary.json)：21组参数的汇总。
- `experiments/*.json`：逐查询延迟、Recall、返回ID、独立加载和线程对照记录。
- Python 脚本：数据校验、索引构建、独立评价和报告生成。
- `manifest.json`：官方开发数据元信息，用于取得同一版本后校验。
- `upload_manifest.json`：本次整理文件的字节数和SHA-256。

## 结果摘要

使用全部1,000,000条1024维向量、1,000条公开query，测试7种配置，每种3档nprobe。当前没有发现同时满足90% Recall和10M外推≤2000MB的配置。

| 配置 | Dev Recall@10 | Dev p95 | 10M空间外推 |
|---|---:|---:|---:|
| IVF256+SQ8，nprobe16 | 97.33% | 19.50ms | 10.32GB |
| PCA256+IVF256+PQ32，全桶扫描 | 59.43% | 16.71ms | 405.78MB |
| IVF256+PQ128，全桶扫描 | 76.05% | 60.80ms | 1362.10MB |

延迟仅为本机 Intel x86/Windows 下的单条索引调用时间，未包含官方协议传输；10M体积为模型外推。未访问鲲鹏官方服务器，未测Final-10M。官方材料中的800MB/1000M/2000MB上限冲突详见报告。

## 复现

原始数据、大型已训练索引和PCA训练中间文件不随此仓库上传。请通过主办方授权渠道获得官方Dev-1M，并将 `base.fbin`、`query.fbin`、`groundtruth.ibin` 放到仓库根目录。代码从公开base训练，query/GT仅供开发评价。

建议创建独立Python环境，参考 `requirements-benchmark.txt` 安装本次测量的依赖。ARM/openEuler安装与编译必须另行验证。

```powershell
python audit.py
python benchmark.py --mode initial
python benchmark.py --mode pq
python run_remaining.py
python final_checks.py
python make_report.py
```

数据SHA校验、21组参数测量和独立加载检查已完成。完整评测方法、笔记本延迟波动、训练强度限制及未执行的官方提交接口检查见报告。这里的索引文件如在复现中生成，只用于Dev实验，不是Final比赛提交包。
