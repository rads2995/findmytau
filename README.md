# findmytau

## Introduction

Risk-Aware Optimal Stopping via Gradient-Boosted Quantile Regression.

## How to Install

### GNU/Linux

Create and activate Python virtual environment:

```bash
python -m venv venv
source ./venv/bin/activate
```

Collect environment variables for building XGBoost and CuPy with HIP support:

```bash
source ./setvars.sh
```

Build and install the application into local build directory:

```bash
python -m pip install -e . -vv
```

Build the XGBoost Python package using the compiled object `thirdparty/xgboost/lib/libxgboost.so`:

```bash
cd ./thirdparty/xgboost/python-package
pip install .
```
## How to Run

To run, simply call the Python application from the terminal:

```bash
findmytau
```
