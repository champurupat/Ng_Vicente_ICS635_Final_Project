# Ng_Vicente_ICS635_Project

1. `/requirements_ubuntu.txt` are requirements for Ubuntu 20.04, CUDA version 12.1, cuDNN version 9.1.0
   - used by tensorflow and pytorch_tabnet libraries
   - speed up training a bit, even with GPU the Optuna search for TabNet took >12 hours
2. Originally, contents of `/data/raw_data` manually retrieved from pro-football-reference.com
    - run `/scripts/preprocessing/merge_data.py` then `/scripts/preprocessing/filter_data.py` to reproduce `data/processed/filtered_data.csv`
3. Run all models Optuna hyperparameter search automatically and sequentially using `/scripts/run_models.py`
    - run models one at a time using scripts in `/scripts/models`
    - change parameters of search in `/scripts/models/constants.py`
4. As Optuna search runs, results are saved in `/results/<model_name>/<model_type>/logs`
5. When respective Optuna search is complete, results are saved in `/results/<model_name>/<model_type>/plotting_data`
6. Run plotting scripts in `/scripts/analysis` to generate plots
    - Individual plots were collated to make figures for report and presentation
    - Figures are saved in `/results/<model_name>/<model_type>/figures`
    - `/scripts/analysis/analyze_data_distribution.py` used to analyze distribution of positions and round labels in the data



