# BPM / BPGT 隐私聚类工具箱

围绕 **Bounded Perturbation Mechanism (BPM)** 与 **BPGT 框架** 搭建的开源实现：提供核心 Python 包、批量实验脚本、历史图表生成工具，便于复现论文 *“K-means clustering with local dχ-privacy *for privacy-preserving data analysis*”* 与 *“BPGT: A Novel Privacy-Preserving K-Means Clustering Framework to Guarantee Local dχ-privacy”* 中的算法，并将其拓展到新的数据集。

---

## 核心功能

- **客户端机制可插拔**：`BPMMechanism`（经典 BPM）与 `BPGMMechanism`（截断指数噪声 + Adam 合成数据）分别实现两种 dχ-privacy 隐私扰动。
- **服务器算法独立**：`KMeansServer`、`GMMServer`、`TMMServer` 在服务器端处理扰动数据；任意机制 × 算法组合都可通过 `PrivacyClusteringPipeline` 组装。
- **BPGT 兼容**：`BPGT` 仅是 `BPGMMechanism + TMMServer` 的组合，保留原论文接口同时支持自由组合。
- **统一实验入口**：`scripts/run_experiment.py` 扫描 ε / L 网格、重复多次取均值，输出 SSE / Silhouette / ARI / NMI 表格并可生成每个机制/算法/L 的折线图。
- **机制诊断 CLI**：`scripts/inspect_mechanism.py` 打印 λ<sub>L</sub>、p<sub>L</sub>、λ<sub>2,r</sub> 等常数，可在 2D 场景下可视化采样分布。
- **历史图表复现**：`scripts/legacy_figures.py`、`scripts/legacy_plots.py` 一键输出旧仓库中的 PNG（如 `iris_evaluation.png`、`kmeans_vs_gmm.png`），方便撰写报告。
- **pytest 覆盖**：`tests/` 内的用例验证噪声采样、BPGM/BPGT 训练及组合式管线，便于在 CI 中自动回归。

---

## BPGT 框架一览

- **BPGM（Bounded Perturbation Generation Mechanism）**  
  按式 (5) 采样截断指数噪声距离 \hat{d}，利用 Adam 最小化  
  `loss = (||r - \hat{r}|| - \hat{d})^2`，生成受限在 `[−L, 1 + L]^d` 的合成数据，保证 dχ-privacy。
- **TK-means**  
  在服务器端使用重尾的 T-Mixture Model + EM（算法 3）对合成数据聚类，获得鲁棒质心并回传标签。

直接调用示例：

```python
from bpm_privacy import BPGT

bpgt = BPGT(
    n_clusters=3,
    epsilon=2.0,
    L=0.5,
    random_state=42,
    gd_lr=0.05,
    gd_tol=1e-3,
    gd_max_iter=200,
)
bpgt.fit(X_normalized)        # X 需提前归一化到 [0,1]^d
labels = bpgt.predict(X_normalized)
print("Private SSE:", bpgt.compute_sse(X_normalized))
```

---

## 安装

### 推荐：uv 工作流

```bash
uv venv                               # 基于 pyproject 创建 .venv
source .venv/bin/activate             # Windows: .\.venv\Scripts\Activate.ps1
uv pip install -e .                   # 安装 bpm_privacy（可编辑模式）
uv pip install -r requirements.txt    # 如需锁定依赖，再安装 requirements
```

之后可使用 `uv run python scripts/run_experiment.py ...` 或 `uv run pytest`。

### 备用：原生 pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

---

## 目录结构

```
.
├── bpm_privacy/
│   ├── mechanism.py          # BPM 数学构件
│   ├── sampling.py           # BPM 采样算法
│   ├── mechanisms.py         # BPMMechanism / BPGMMechanism
│   ├── server_algorithms.py  # KMeansServer / GMMServer / TMMServer
│   ├── pipeline.py           # PrivacyClusteringPipeline（机制 × 服务器）
│   ├── private_tmm.py        # TMM 数学实现（服务器端使用）
│   └── bpgt.py               # BPGM 核心 + BPGT 兼容封装
├── scripts/
│   ├── run_experiment.py     # 批量实验 CLI（含折线图）
│   ├── inspect_mechanism.py  # 机制常数与采样诊断
│   ├── legacy_figures.py     # 生成历史 PNG
│   └── legacy_plots.py       # 经典 Iris 指标曲线
├── tests/                    # pytest 用例
├── figures/、figures-exp/    # 示例图输出
├── pyproject.toml  /  requirements.txt  /  uv.lock
└── graduate.egg-info         # 可编辑安装元数据
```

---

## 快速开始

### 1. 批量实验 + 绘图

```bash
uv run scripts/run_experiment.py \
  --dataset iris \
  --mechanisms bpm bpgm \
  --servers kmeans gmm tmm \
  --epsilons 0.5 1 2 4 \
  --Ls 0.3 0.5 \
  --trials 5 \
  --csv results.csv \
  --plot figures-exp/full-exp.png
```

常用参数（含枚举值）：

| 参数 | 说明 |
|------|------|
| `--dataset {iris, blobs}` | 选择真实或合成数据集。 |
| `--mechanisms {bpm, bpgm}` | 选择客户端隐私机制（可多选，逐一组合）。 |
| `--servers {kmeans, gmm, tmm}` | 选择服务器端聚类算法（可多选，逐一组合）。 |
| `--epsilons`, `--Ls` | 输入若干 ε / L 组合衡量隐私-效用。 |
| `--trials N` | 每组配置重复 N 次求平均。 |
| `--skip-baseline` | 仅跑隐私组合时开启。 |
| `--csv path` | 输出结果表格。 |
| `--plot path` | 保存 SSE / Silhouette / ARI / NMI vs ε 的折线图（按 Mechanism+Server+L 区分）。 |

其他常用参数：`--samples`、`--features`、`--clusters`、`--seed`、`--tmm-*`（控制 TMMServer）。  
若选择 `bpgm`，还可设置 `--bpgt-gd-lr`、`--bpgt-gd-tol`、`--bpgt-gd-max-iter` 控制合成数据的 Adam 步长/迭代数。

### 2. 检查 BPM / BPGM 机制

```bash
python scripts/inspect_mechanism.py \
  --epsilon 2.0 \
  --L 0.3 \
  --dimension 2 \
  --samples 2000 \
  --plot figures/bpm_scatter.png
```

打印 λ<sub>L</sub>、p<sub>L</sub>、λ<sub>2,r</sub> 并在 2D 情况下生成采样散点图。

### 3. 重建历史图表

```bash
python scripts/legacy_figures.py           # 生成所有旧 PNG
python scripts/legacy_plots.py             # Iris 指标曲线与聚类散点
python scripts/legacy_figures.py --figures kmeans_vs_gmm nonlinear_analysis
```

图像保存在 `figures/`，命名保持与旧仓库一致（如 `iris_evaluation.png`、`all_methods_comparison.png`）。

---

## 测试

```bash
uv run pytest    # 或 python -m pytest
```

- `tests/test_mechanism.py`：验证 BPM 密度单调、采样范围及 p<sub>L</sub> 的经验命中率。
- `tests/test_private_kmeans.py`：确保扰动数据位于报告空间，并测试 ε 增大后的 SSE 改善趋势。
- `tests/test_bpgt.py`：检查 BPGM 的噪声距离与合成记录范围，以及 BPGT 的端到端训练。

---

## 引用

如在研究或产品中使用此实现，请引用以下论文并（可选）附上仓库链接：

> Fan Chen et al. *BPGT: A Novel Privacy-Preserving K-Means Clustering Framework to Guarantee Local dχ-privacy*, 2025.

---

## 许可与声明

本项目面向研究与实验用途。请遵守所在地区的隐私法规，并对 BPGT 的原作者给予适当致谢。如需展示基于本工具箱的成果或有任何疑问，欢迎在 Issues 中讨论。 
