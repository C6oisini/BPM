# BPGT（d_privacy）：距离隐私聚类工具箱

`d_privacy` 是一套面向局部隐私（Local Privacy）场景的聚类工具。它包含四种客户端机制（BPM、BPGM、BLM、CIM）、三种服务器端聚类算法（KMeans、GMM、TMM）以及可组合的 `PrivacyClusteringPipeline`。通过 `scripts/run_experiment.py` 可以批量扫描 ε（及需要时的 L）并输出 CSV / 折线图，`scripts/inspect_mechanism.py` 则可视化 BPGM 的采样区域。

## 主要特性

- **机制可插拔**：  
  - `BPMMechanism`（经典 BPM，报告空间由用户指定 `L`）。  
  - `BPGMMechanism`（BPGT 里的截断指数噪声 + Adam 合成数据，同样需要 `L`）。  
  - `BLMMechanism`（自动计算数据集中任意两点的最大距离并作为噪声上限，不再使用 L）。  
  - `CIMMechanism`（噪声支撑固定为 [0,1]，仅受 ε 控制）。  

- **服务器端自由组合**：`KMeansServer`、`GMMServer`、`TMMServer` 都可接入 `PrivacyClusteringPipeline`，可实现任意机制 × 服务器的实验矩阵。

- **实验脚本**：`scripts/run_experiment.py` 支持：
  - 批量扫描 ε / L 网格（仅 BPM/BPGM 使用 L）。
  - 输出 SSE / Silhouette / ARI / NMI / RE 指标。
  - 生成 ε–性能折线图（SVG/PNG）。

- **BPGM 采样可视化**：`scripts/inspect_mechanism.py` 打印 BPGM 配置、距离统计，并在 2D 时绘出采样散点。

## 安装

建议使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt    # 若需固定版本
```

若使用原生 pip：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## 目录结构

```
.
├── d_privacy/
│   ├── client.py      # BPM/BPGM/BLM/CIM 实现
│   ├── server.py      # KMeans/GMM/TMM 服务器及 TMM 数学实现
│   └── pipeline.py    # PrivacyClusteringPipeline + BPGT 封装
├── scripts/
│   ├── run_experiment.py
│   └── inspect_mechanism.py
├── pyproject.toml / requirements.txt / uv.lock
```

## 快速开始

### 批量实验

以 iris 数据集为例，运行 BPM + BPGM + BLM + CIM，KMeans 服务器，并输出 CSV / 图像：

```bash
uv run scripts/run_experiment.py \
  --dataset iris \
  --mechanisms bpm bpgm blm cim \
  --servers kmeans \
  --epsilons 0.2 0.5 1 2 4 \
  --Ls 0.2 0.4 \
  --trials 3 \
  --csv iris_results.csv \
  --plot figures-exp/iris_results.png
```

说明：

| 参数 | 作用 |
|------|------|
| `--dataset {iris, blobs}` | 选择内置数据集。 |
| `--mechanisms ...` | 客户端机制，`bpm/bpgm` 需要 `--Ls`，`blm/cim` 会忽略 `--Ls`。 |
| `--servers ...` | 可选 `kmeans` / `gmm` / `tmm`。 |
| `--epsilons ...` | ε 网格。 |
| `--Ls ...` | 仅 BPM/BPGM 使用的报告空间参数；BLM 自动计算最大距离，CIM 固定支撑。 |
| `--trials` | 每组配置重复次数。 |
| `--csv` / `--plot` | 输出表格和折线图。 |

附加参数：

- BPGM：`--bpgt-gd-lr`、`--bpgt-gd-tol`、`--bpgt-gd-max-iter`（控制 Adam）。
- BLM / CIM：`--distance-lr`、`--distance-tol`、`--distance-max-iter`（合成数据的梯度下降）。
- TMM：`--tmm-*` 诸参数控制服务器端 EM。

> 注：`BLMMechanism` 会在运行时计算归一化后数据的最大 pairwise 距离并作为噪声上限；`CIMMechanism` 不含 L，也不提供额外的支撑参数，噪声距离始终固定在 [0,1]。

### BPGM 采样诊断

```bash
uv run scripts/inspect_mechanism.py \
  --epsilon 2.0 \
  --L 0.5 \
  --dimension 2 \
  --samples 2000 \
  --plot figures/bpgm_sampling.png
```

脚本会打印 BPGM 配置、样本距离、球内/报告空间命中率，并在 2D 情况下生成散点图（可用于分析合成点范围）。

## 实验输出

`run_experiment.py` 的 CSV 包含以下列：

- `mode`（baseline/private）、`mechanism`、`server`、`epsilon`、`L`（对无 L 机制设为 `-`）。
- 指标：`SSE`、`Silhouette`、`ARI`、`NMI`、`RE`（质心相对误差，只有数据集自带标签时才有效）。

折线图会针对每个服务器绘制多条曲线，横轴为 ε，纵轴为相应指标，并将四种机制放在一张图中便于对比。

## 测试

```bash
uv run pytest
```

覆盖度集中在噪声采样、BPGM/BPGT 流程以及组合式管线。根据需要可扩展更多数据集或服务器实现。

## Follow us

原文：Chen et al. 2025. “BPGT: A Novel Privacy-Preserving K-Means Clustering Framework to Guarantee Local dχ-Privacy.” Preprint.

👏 欢迎提出问题、提交Issue或发起Pull Request！

