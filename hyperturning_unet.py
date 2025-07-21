from netCDF4 import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import recall_score
import keras_tuner as kt
import tensorflow as tf
import json

# Local imports
from utils import zscore, with_channel, cropping, unet_regressor
from metrics import RecallMetric



# Load predictor datasets
evv_xr   = xr.open_dataset("predictor/AFR_1984-2023_daily_evv_rewrite.nc")
msl_xr   = xr.open_dataset("predictor/AFR_1984-2023_daily_msl_rewrite.nc")
pr_xr    = xr.open_dataset("predictor/AFR_1984-2023_daily_pr_rewrite.nc")
snsr_xr  = xr.open_dataset("predictor/AFR_1984-2023_daily_snsr_rewrite.nc")
tcc_xr   = xr.open_dataset("predictor/AFR_1984-2023_daily_tcc_rewrite.nc")
u10m_xr  = xr.open_dataset("predictor/AFR_1984-2023_daily_u10m_rewrite.nc")
vswl1_xr = xr.open_dataset("predictor/AFR_1984-2023_daily_vswl1_rewrite.nc")
vswl2_xr = xr.open_dataset("predictor/AFR_1984-2023_daily_vswl2_rewrite.nc")
z500_xr  = xr.open_dataset("predictor/AFR_1984-2023_daily_z500_rewrite.nc")

# Normalize and add channel dimension
evv_anom   = with_channel(zscore(cropping(evv_xr)["e"]),       "evv")
msl_anom   = with_channel(zscore(cropping(msl_xr)["msl"]),     "msl")
pr_anom    = with_channel(zscore(cropping(pr_xr)["pr"]),       "pr")
snsr_anom  = with_channel(zscore(cropping(snsr_xr)["ssr"]),    "snsr")
tcc_anom   = with_channel(zscore(cropping(tcc_xr)["tcc"]),     "tcc")
u10m_anom  = with_channel(zscore(cropping(u10m_xr)["u10"]),    "u10m")
vswl1_anom = with_channel(zscore(cropping(vswl1_xr)["swvl1"]), "vswl1")
vswl2_anom = with_channel(zscore(cropping(vswl2_xr)["swvl2"]), "vswl2")
z_anom     = with_channel(zscore(cropping(z500_xr)["z"]),      "z500")

# Combine into one DataArray
predictors = xr.concat([z_anom, tcc_anom, snsr_anom, evv_anom], dim="channel")
X = predictors.values

# Load target datasets
target  = xr.open_dataset("target_var/AFR_1984-2023_daily_tw_heatwaves_day.nc")
target = target.where(target.time.dt.year.isin(np.arange(1984, 2024)), drop=True)
y = with_channel(cropping(target)["tw_heatwaves_day"], "label").values

# Compute class weights from full pixel set
y_flat = y.reshape(-1)
count_0 = np.sum(y_flat == 0)
count_1 = np.sum(y_flat == 1)
total = count_0 + count_1
weight_0 = total / (2 * count_0)
weight_1 = total / (2 * count_1)

# Weighted loss function
def weighted_bce(weight_0, weight_1):
    def loss(y_true, y_pred):
        return -K.mean(weight_1 * y_true * K.log(y_pred + 1e-7) +
                       weight_0 * (1 - y_true) * K.log(1 - y_pred + 1e-7))
    return loss

loss_fn = weighted_bce(weight_0, weight_1)


# Define the hypermodel
def build_model(hp):
    # Tune hyperparameters
    n_filters = hp.Choice("n_filters", [16, 32, 64])
    dropout_rate = hp.Float("dropout_rate", 0.2, 0.5, step=0.1)
    l2_strength = hp.Choice("l2_strength", values=[1e-2, 1e-3, 1e-4, 1e-5])
    lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Choice("batch_size", [16, 32, 64, 128, 256, 512])

    # Build model using the unet_regressor
    model = unet_regressor(
        input_shape=(320, 320, 4),
        n_filters=n_filters,
        dropout_rate=dropout_rate,
        l2_strength=l2_strength,
        lr=lr,
	loss_fn=loss_fn,
        metrics=[RecallMetric()]
    )

    return model

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_recall", direction="max"),
    max_trials=20,
    executions_per_trial=1,
    overwrite=True,
    project_name="unet_heatwave_tuning"
)

# Start the search
tuner.search(
    X, y,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model
best_model.save("best_unet_heatwave_model.keras")

# Print best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print("Best hyperparameters:")
print(best_hps.values)
