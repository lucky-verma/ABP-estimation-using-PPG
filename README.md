# ABP Estimation Using PPG

A machine learning pipeline for estimating Arterial Blood Pressure (ABP) waveforms from Photoplethysmogram (PPG) signals using deep learning and time series forecasting approaches.

## Overview

This repository implements multiple approaches to predict blood pressure from PPG signals:
- **PPG2ABP**: Fully Convolutional Neural Networks (U-Net, MultiResUNet) for signal-to-signal translation
- **Linear Regression & CNN**: Traditional ML and neural network baselines
- **Time Series Forecasting**: PyCaret-based ARIMA and AutoML approaches

## Repository Structure

```
ABP-estimation-using-PPG/
├── PPG2ABP/                        # PPG to ABP signal translation module
│   ├── codes_125hz/                # Implementation for 125Hz sampling rate
│   │   ├── models.py               # U-Net, MultiResUNet, and other architectures
│   │   ├── train_models.py         # Training scripts
│   │   ├── evaluate.py             # Evaluation and metrics
│   │   ├── data_processing.py      # Data preprocessing utilities
│   │   ├── streamlit-app.py        # Web demo application
│   │   └── PPG2ABP.ipynb           # Demo notebook
│   ├── codes_25hz/                 # Implementation for 25Hz sampling rate
│   ├── weights/                    # Pre-trained model weights
│   └── requirements.txt            # PPG2ABP-specific dependencies
├── tanzen_data/                    # Data scraping utilities
│   └── data_scrap.ipynb            # Data collection notebook
├── linear_cnn_pycaret.ipynb        # Linear regression, CNN, and PyCaret regression
├── arima_pycaret.ipynb             # Time series forecasting with PyCaret
├── requirements.txt                # Main project dependencies
└── CODE_OF_CONDUCT.md
```

## Tech Stack

- **Languages**: Python 3.8+
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **ML/AutoML**: PyCaret, scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web App**: Streamlit, Gradio

## Installation

```bash
# Clone the repository
git clone git@github.com:lucky-verma/ABP-estimation-using-PPG.git
cd ABP-estimation-using-PPG

# Install dependencies
pip install -r requirements.txt

# For PPG2ABP module specifically
pip install -r PPG2ABP/requirements.txt
```

## Usage

### PPG2ABP Signal Translation

```bash
cd PPG2ABP/codes_125hz
python train_models.py      # Train models
python evaluate.py          # Evaluate on test set
streamlit run streamlit-app.py  # Launch web demo
```

### Jupyter Notebooks

1. **linear_cnn_pycaret.ipynb**: Baseline models (Linear Regression, CNN) and PyCaret AutoML
2. **arima_pycaret.ipynb**: Time series forecasting approaches
3. **PPG2ABP/codes_125hz/PPG2ABP.ipynb**: Full PPG2ABP pipeline demo

## Data

The project uses PPG and ABP signal data from MATLAB (.mat) files. Data should be placed in a `kaggle_data/` directory with files named `part_1.mat` through `part_12.mat`.

## Citation

If you use PPG2ABP in your research, please cite:

```bibtex
@article{ibtehaz2020ppg2abp,
  title={PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks},
  author={Ibtehaz, Nabil and Rahman, M Sohel},
  journal={arXiv preprint arXiv:2005.01669},
  year={2020}
}
```

## License

MIT License
