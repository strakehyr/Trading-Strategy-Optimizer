# Modular Algorithmic Trading & Optimization Framework

> **A strategy-agnostic ecosystem for rigorous backtesting, robust parameter optimization, and regime analysis.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-Technical%20Guide-green)](docs/TECHNICAL_GUIDE.md)

## 📖 Overview
This repository contains a framework designed to bridge the gap between theoretical trading strategies and market reality. Unlike standard backtesters that prioritize raw returns, this system prioritizes **statistical robustness**.

It helps researchers answer the critical question: *"Is this strategy actually good, or did I just overfit the parameters to historical noise?"*

### Core Philosophy
1.  **Strict IS/OOS Separation:** Never optimize on the data you validate with.
2.  **Degradation over Profit:** The best parameter set is not the one with the highest return, but the one with the least performance degradation between training and testing.
3.  **Regime Awareness:** Strategies are stress-tested against specific market conditions (Bull, Bear, Sideways).

---

## 🚀 Key Capabilities

*   **CMA-ES Optimization:** Uses the [Covariance Matrix Adaptation Evolution Strategy](https://arxiv.org/abs/1604.00772) (via Optuna) to navigate non-convex parameter spaces without gradients.
*   **Vectorized Simulation:** High-performance engine handling entry/exit logic, position sizing, and commission modeling.
*   **Modular Architecture:** "Plug-and-play" design allows you to drop new Python strategy files into a folder and immediately start optimizing them.
*   **Robustness Metrics:** Automatically calculates Sharpe, Calmar, and Degradation coefficients.

---

## 📊 Analytics & Visualization

The framework generates a comprehensive suite of interactive HTML reports for every simulation:

| Report | Description |
| :--- | :--- |
| **Backtest Performance** | Interactive candlestick charts with overlay indicators, trade markers, and equity curves. |
| **Robustness Scatter** | A correlation view of In-Sample vs. Out-of-Sample metrics to identify overfitting visually. |
| **Regime Matrix** | A breakdown of strategy performance during specific market regimes (e.g., "Bullish" vs. "Sideways"). |
| **Parallel Coordinates** | A visualization of the hyperparameter search space, highlighting high-performance clusters. |
| **Distribution Boxplots** | Visualizes the variance of returns across *all* trial iterations to distinguish alpha from luck. |
| **Stability Analysis** | Bar charts comparing In-Sample vs. Out-of-Sample metrics to detect immediate performance decay. |

---

## 🛠️ Getting Started

For detailed instructions on installation, data ingestion (API or CSV), and creating your own strategies, please refer to the technical documentation. (docs/TECHNICAL_GUIDE.md)

### Quick Start (Pre-requisites)
```bash
git clone https://github.com/strakehyr/Trading-Strategy-Optimizer.git
pip install -r requirements.txt
python main.py --symbols QQQ --strategies all
