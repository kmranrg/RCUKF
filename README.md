# RCUKF: Reservoir Computing with Unscented Kalman Filter

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)

**RCUKF** is a hybrid framework that combines **Reservoir Computing (RC)** and the **Unscented Kalman Filter (UKF)** for robust, data-driven state estimation in nonlinear dynamical systems. This approach avoids explicit modeling of system dynamics and is especially powerful for chaotic and partially observed systems.

---

## 📘 Features

- ✅ Model-free learning using Reservoir Computing (Echo State Network)
- ✅ Online filtering with Unscented Kalman Filter
- ✅ No backpropagation: fast training using ridge regression
- ✅ Supports chaotic systems like Lorenz, Rössler, Mackey-Glass
- ✅ Plug-and-play modular design

---

## 🗂️ Project Structure

RCUKF/  
├── benchmarks/ # RC vs RCUKF RMSE comparison and plots  
├── data_gen/ # Noisy synthetic data generators  
├── demos/ # End-to-end implementations of RCUKF  
├── rcukfpy/ # Core implementation (RC, UKF, RCUKF)  
├── utils/ # RMSE calculator and helper utils  
├── LICENSE  
└── README.md


---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rcukf.git
cd rcukf
```

### 2. Install dependencies
```bash
pip install numpy matplotlib
```

## 🔬 Usage

### 🔁 Run a benchmark
Compare RC-only vs RC+UKF on the Lorenz system:
```bash
python -m benchmarks.RC_vs_RCplusUKF_Lorenz
```

Other systems:
```bash
python -m benchmarks.RC_vs_RCplusUKF_Rossler
python -m benchmarks.RC_vs_RCplusUKF_MackeyGlass
```

### 🧪 Run an end-to-end demo
```bash
python demos.RCplusUKF_LorenzImplementation
python demos.RCplusUKF_MackeyGlassImplementation
python demos.RCplusUKF_RosslerImplementation
```

## 🧠 How It Works
- `ReservoirComputer`: Trains a readout model to approximate system dynamics.
- `UnscentedKalmanFilter`: Uses sigma points to filter noisy observations.
- `RC_UKF`: Wraps both modules to predict system state using data + measurements.

## 📁 Module Overview

| Module                   | Description                                          |
| ------------------------ | ---------------------------------------------------- |
| `rcukfpy/RC.py`          | Echo State Network (Reservoir Computer)              |
| `rcukfpy/UKF.py`         | UKF implementation for nonlinear state estimation    |
| `rcukfpy/RCplusUKF.py`   | Combines RC and UKF into a hybrid state estimator    |
| `data_gen/*.py`          | Noisy time series generators (Lorenz, Rossler, etc.) |
| `utils/compute_error.py` | RMSE computation per dimension                       |
| `benchmarks/*.py`        | Compare RMSE of RC-only vs RC+UKF                    |
| `demos/*.py`             | Full pipeline demos (training + filtering + plots)   |

## 🤝 Contributing
Pull requests are welcome! Please open an issue first to discuss any major changes or improvements.

> Developed with ❤️ by Kumar Anurag
[Website](https://kmranrg.com) | [LinkedIn](https://linkedin.com/in/kmranrg)







