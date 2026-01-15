# TrueInflowToolkit-TIT-
# SWMM-Kalman Inflow Reconstruction

A Python toolbox for upstream inflow reconstruction in long river reaches / river-type reservoirs by integrating a hydrodynamic model with a Kalman smoother. It is particularly advantageous when the available water-level observations are far from the upstream inflow section.

## Overview
Estimating the “true” upstream inflow is challenging when (i) direct inflow measurements are unavailable and (ii) stage observations are limited and located far downstream. This project couples a 1D hydrodynamic routing model with a Kalman smoother to infer upstream inflow that best matches observed water levels while respecting hydraulic dynamics.

## Key Features
- Designed for **long river reaches and river-type reservoirs**, especially when observation stations are **far from the upstream boundary**.
- Fully implemented in **Python** with **easy-to-run scripts**.
- Supports **CPU acceleration / multiprocessing** for faster computation (please set a safe number of workers).

## Requirements
- Python 3.x
- Common scientific Python stack (e.g., `numpy`, `scipy`, `pandas`, `matplotlib`), plus any other dependencies listed in `requirements.txt` (if provided).

## Input Data
This case study requires:
1. **River geometry / terrain data** (cross-sections or channel geometry required by the hydrodynamic model)
2. **Water level observations** (stage time series)
3. **Downstream outflow (release) data** (time series at the outlet)

Data units and time step should be consistent across all inputs.

## Quick Start
1. Put **all input files** and the **code** in the **same directory** (as provided in this repository).
2. Run:
   ```bash
   python swmmkalman.py
   python plotcase.py

## Configuration (Key Parameters)

The main parameters are defined in `obstest/par.txt` (or the config section it reads). Typical meanings:

- `hottime` (h): warm-up data length.
- `simtime` (h): warm-up simulation time.
- `stabletime` (h): stabilization period during warm-up.
- `waitingtime`: waiting time offset (if applicable).
- `listlaterflow`: flow-inlet section IDs for which warm-up stabilization is applied (list).
- `n`: ensemble size.
- `u`: prior standard deviation(s) of inflow to be estimated. For upstream inflow inversion, only the first element is effective.
- `q`: initial inflow(s) at the first time step. For upstream inflow inversion, only the first element is effective.
- `lentime` (h): length of historical inflow used as input (lag window).
- `numcores`: number of CPU processes for parallel computation.  
  **Warning:** set this according to your machine to avoid freezing.
- `RTStime`: backward smoothing window for RTS (hours).
- `numflowava`: number of inflow time steps estimated simultaneously (recommended `1`).
- `numheadava`: number of water-level observations used simultaneously (recommended `3`).
- `numdeleteheadb`: number of upstream stage stations excluded (for sensitivity/comparison experiments).
- `numdeleteheadf`: number of downstream stage stations excluded (for sensitivity/comparison experiments).
- `dq`: threshold of stage fluctuation used in estimating inflow propagation lag time.
- `Iddetet`: IDs of intermediate stage stations to exclude (list; for comparison experiments).
- `option`: `True` = Ensemble Kalman Smoother; `False` = Ensemble Kalman Filter.

## Note on Hydrodynamic Model
Note: The hydrodynamic model component is not included in this public release due to confidentiality constraints related to terrain data. If you require the complete version with the hydrodynamic model for research purposes, please contact the corresponding author: liupan@whu.edu.cn.

## Results and Visualization
We have uploaded all calculation results in the plot folder and provided plotting code examples. You can reproduce the figures from the paper:

Fig.8 (Section 4.2.3): Total inflow reconstruction results
Fig.6(c,d) (Section 4.2.2): Sensitivity analysis of observation configurations
To generate these figures, run the provided plotting scripts or modify them according to your needs.
