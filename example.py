import sys
sys.path.append('/mnt/c/Users/Алия/darts')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import EnergyDataset
from darts.models import RNNModel
from darts.timeseries import concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import logging

logging.disable(logging.CRITICAL)


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

length = 400
trend = tg.linear_timeseries(length=length, end_value=4)
season1 = tg.sine_timeseries(length=length, value_frequency=0.05, value_amplitude=1.0)
noise = tg.gaussian_timeseries(length=length, std=0.6)
noise_modulator = (
    tg.sine_timeseries(length=length, value_frequency=0.02)
    + tg.constant_timeseries(length=length, value=1)
) / 2
noise = noise * noise_modulator

target_series = sum([noise, season1])
covariates = noise_modulator
target_train, target_val = target_series.split_after(0.65)

print(f"init RNN model in example.py")
my_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=50,
    optimizer_kwargs={"lr": 1e-3},
    random_state=0,
    training_length=50,
    input_chunk_length=20,
    likelihood=GaussianLikelihood(),
    **generate_torch_kwargs(),
)

my_model.fit(target_train, future_covariates=covariates)