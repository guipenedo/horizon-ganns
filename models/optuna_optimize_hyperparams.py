"""
    Example optuna script to optimize the model's hyperparameters
    (seeks to maximize F1-Score)
"""

import optuna
from models import rcgan
from utils.pando_utils import OPTUNA_PATH


def objective(trial):
    params = {
        'lr_g': trial.suggest_loguniform('lr_g', 1e-5, 1e-1),
        'lr_d': trial.suggest_loguniform('lr_d', 1e-5, 1e-1),
        'l2_loss_weight': trial.suggest_loguniform('l2_loss_weight', 1e-4, 0.5),
        'variety_loss_k': trial.suggest_int('variety_loss_k', 1, 5),
        'num_epochs': 20
    }
    dargs = vars(rcgan.opt)
    dargs.update(params)
    return rcgan.train(save=False)


study = optuna.create_study(study_name=rcgan.MODEL_NAME, direction='maximize',
                            load_if_exists=True, storage=f'sqlite:///{OPTUNA_PATH}')
study.optimize(objective, n_trials=100)
print(study.best_params)
