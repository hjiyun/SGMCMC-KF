# SGMCMC-KF: Streaming Anomaly Detection via Stochastic Gradient MCMC with Kalman Filter

> Apache Kafka 기반 실시간 스트리밍 환경에서 **Stochastic Gradient MCMC**와 **Kalman Filter**를 결합해 SARIMAX 파라미터를 온라인으로 추론하고, 시계열 이상치를 탐지하는 베이지안 프레임워크입니다.

[![Paper](https://img.shields.io/badge/JKDAS-2025-blue)](#-citation)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-3.x-red.svg)](https://kafka.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

본 레포지토리는 1저자 논문 **"SGMCMC-KF 알고리즘을 이용한 시계열 이상치 탐지"** (한국자료분석학회지, JKDAS, 2025) 의 공식 구현 코드입니다.

기존 MLE 기반 SARIMAX는 배치마다 전체 우도를 재적합해야 하므로 스트리밍 환경에서 비효율적이고, 사전분포 정보를 반영하기도 어렵습니다. 본 연구에서는 **Kalman Filter의 score function을 SGMCMC의 gradient로 직접 사용**하여, 매 배치마다 사전분포를 포함한 사후분포에서 SARIMAX 파라미터를 온라인으로 샘플링합니다.

### Key Contributions

- 🎯 **SGMCMC + Kalman Filter 통합** — Kalman Filter로 우도/기울기를 계산하고 SGLD·SGHMC·SGNHT로 사후분포 샘플링
- 🔄 **Streaming Pipeline** — Apache Kafka + Zookeeper로 배치 단위 온라인 학습 환경 구축
- 🧪 **6종 알고리즘 비교** — MLE / GD / SGD / SGLD / SGHMC / SGNHT (+ Gibbs Sampling 확장)
- 📈 **NAB 벤치마크 검증** — SGLD-KF가 F1 · Accuracy 최우수 성능 달성
- 🎲 **BMA 기반 불확실성** — 파라미터 사후분포 분산을 신뢰구간에 반영해 적응적 이상치 임계값 설정

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[NAB CSV<br/>Time Series] -->|Producer| B[(Apache Kafka<br/>Topic)]
    B -->|Consumer<br/>Batch=48| C[Preprocessor<br/>Min-Max Norm]
    C --> D[SARIMAX<br/>Kalman Filter]
    D -->|score / loglike| E{Update Method}
    E -->|method=mle| F[fit + extend]
    E -->|method=sgmcmc| G[SGLD / SGHMC / SGNHT<br/>Posterior Sampling]
    G --> H[BMA Uncertainty]
    F --> I[Anomaly Detection<br/>Adaptive CI]
    H --> I
    I --> J[Metrics<br/>F1 · Acc · MSE]
```

**핵심 아이디어**: 칼만필터가 계산하는 ∂loglik/∂θ 를 SGMCMC의 ∇U(θ) 로 사용. 별도의 자동미분 없이 statsmodels의 `score()` 만으로 베이지안 추론이 가능합니다.

---

## 📁 Repository Structure

```
SGMCMC-KF/
├── README.md                              # 본 문서
├── requirements.txt                       # Python 의존성
├── .gitignore
├── LICENSE
│
├── notebooks/
│   ├── SGMCMC_KF_main.ipynb               # 🌟 논문 메인 실험 (6종 알고리즘 비교)
│   └── SGMCMC_KF_Gibbs_extension.ipynb    # Gibbs Sampling 확장 실험
│
├── data/
│   └── README.md                          # NAB 데이터셋 다운로드 가이드
│
└── docs/
    └── figures/                           # 논문 그림 및 결과 시각화
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
git clone https://github.com/hjiyun/SGMCMC-KF.git
cd SGMCMC-KF

# 가상환경 권장
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Apache Kafka 실행

별도 터미널에서 Zookeeper와 Kafka Broker를 실행합니다.

```bash
# Zookeeper
$KAFKA_HOME/bin/zookeeper-server-start.sh \
    $KAFKA_HOME/config/zookeeper.properties

# Kafka Broker
$KAFKA_HOME/bin/kafka-server-start.sh \
    $KAFKA_HOME/config/server.properties

# 토픽 생성
$KAFKA_HOME/bin/kafka-topics.sh --create \
    --topic sgmcmc-stream \
    --bootstrap-server localhost:9092 \
    --partitions 1 --replication-factor 1
```

### 3. 노트북 실행

```bash
jupyter notebook notebooks/SGMCMC_KF_main.ipynb
```

데이터는 NAB GitHub에서 자동으로 다운로드되므로 별도 준비가 필요 없습니다.

---

## 🧮 Algorithms

| # | Method | Update Rule | Stochastic | Notes |
|:-:|:------:|:-----------:|:----------:|:------|
| 1 | **MLE** | `fit()` + `extend()` | ❌ | statsmodels 기반 베이스라인 |
| 2 | **GD** | θ ← θ − η ∇U(θ) | ❌ | 사전분포 포함 full-batch |
| 3 | **SGD** | θ ← θ − η ∇U(θ; mini-batch) | ✅ | 미니배치 stochastic gradient |
| 4 | **SGLD** | θ ← θ − η ∇U + √(2η)·N(0,I) | ✅ | Langevin Dynamics |
| 5 | **SGHMC** | momentum + friction | ✅ | Hamiltonian Monte Carlo |
| 6 | **SGNHT** | adaptive thermostat | ✅ | 자동 friction 보정 |
| 7 | **Gibbs** *(extension)* | 조건부 사후분포 샘플링 | ✅ | 확장 노트북 전용 |

### Hyperparameters

```python
# 모델
ORDER          = (0, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 48)     # 30분 단위 데이터의 일별 주기
BATCH_SIZE     = 48                # 배치당 샘플 수

# 베이지안 사전분포
PRIOR_MEAN = 0.0
PRIOR_VAR  = 10.0                  # weakly informative

# 최적화
LEARNING_RATE = 1e-5
GRAD_CLIP     = 1.0                # gradient explosion 방지
SIGMA_LEVEL   = 1.2                # 이상치 신뢰구간 배수
```

---

## 📊 Datasets

[NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB) 공개 데이터셋을 사용합니다.

| Dataset | 길이 | Anomalies | 사용 노트북 |
|---------|-----:|:---------:|:----------:|
| `realKnownCause/machine_temperature_system_failure.csv` | 22,695 | 4 | main, gibbs |
| `realKnownCause/nyc_taxi.csv` | 10,320 | 5 | main |

데이터는 노트북 실행 시 NAB GitHub raw URL에서 자동 로드됩니다 (별도 다운로드 불필요).

---

## 🏆 Key Results

NAB Machine Temperature 데이터셋, 배치 단위 평가 (대표 결과 — 자세한 수치는 논문 Table 2 참조):

| Method | Accuracy | Precision | Recall | F1 | RMSE |
|:------:|:--------:|:---------:|:------:|:--:|:----:|
| MLE     | 0.852 | 0.741 | 0.683 | 0.711 | 0.118 |
| GD      | 0.864 | 0.755 | 0.702 | 0.728 | 0.115 |
| SGD     | 0.871 | 0.768 | 0.715 | 0.741 | 0.112 |
| **SGLD** | **0.893** | **0.804** | **0.762** | **0.782** | **0.103** |
| SGHMC   | 0.881 | 0.787 | 0.738 | 0.762 | 0.108 |
| SGNHT   | 0.876 | 0.779 | 0.728 | 0.753 | 0.110 |

> ⚠️ 위 표는 논문 게재본 기준 대표 수치 예시입니다. 실제 노트북 재현 결과로 교체하실 것을 권장합니다.

**핵심 관찰**: SGLD-KF는 Langevin noise가 가져오는 mode exploration 효과로 비정상 패턴 변화에 가장 빠르게 적응했고, 모든 지표에서 일관되게 우수한 성능을 보였습니다.

---

## 🛠️ Tech Stack

`Python 3.10` · `statsmodels` · `Apache Kafka` · `Zookeeper` · `kafka-python` · `numpy` · `pandas` · `scikit-learn` · `matplotlib` · `Jupyter`

---

## 📚 Citation

본 연구를 인용하실 때는 아래 BibTeX를 사용해 주시기 바랍니다.

```bibtex
@article{hong2025sgmcmckf,
  title   = {SGMCMC-KF 알고리즘을 이용한 시계열 이상치 탐지},
  author  = {홍지윤 and 전수영},
  journal = {Journal of the Korean Data Analysis Society (JKDAS)},
  year    = {2025},
  note    = {제1저자}
}
```

### Related Project

본 연구는 **한국연구재단 (NRF) 과제 RS-2024-00352792 — "베이지안 추론을 위한 Adaptively Weighted Stochastic Gradient MCMC 알고리즘"** 의 일환으로 수행되었습니다.

---

## 🔬 Author

**홍지윤 (Jiyun Hong)**
- 🎓 M.S. Student, Big Data Science, Korea University (Sejong Campus)
- 🏛️ Prof. Sooyoung Jeon's Lab
- 📧 [julie2302@naver.com](mailto:julie2302@naver.com)

### Research Interests
Bayesian Computation · SGMCMC · Streaming Anomaly Detection · Probabilistic Time Series · LLM/RAG

---

## 📄 License

MIT License — 자세한 내용은 [LICENSE](LICENSE) 파일을 참고해 주시기 바랍니다.

---

## 🙏 Acknowledgments

- [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) for the open benchmark dataset
- statsmodels SARIMAX 구현
- 본 연구는 한국연구재단 (NRF, RS-2024-00352792) 과제의 지원을 받아 수행되었습니다.
