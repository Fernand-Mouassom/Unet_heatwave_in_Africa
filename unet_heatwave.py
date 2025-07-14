from netCDF4 import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import make_scorer, recall_score
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import BaseCrossValidator
from tensorflow.keras import backend as K
import json
from utils import zscore, with_channel, unet_regressor

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
evv_anom   = with_channel(zscore(evv_xr["e"]),       "evv")
msl_anom   = with_channel(zscore(msl_xr["msl"]),     "msl")
pr_anom    = with_channel(zscore(pr_xr["pr"]),       "pr")
snsr_anom  = with_channel(zscore(snsr_xr["ssr"]),    "snsr")
tcc_anom   = with_channel(zscore(tcc_xr["tcc"]),     "tcc")
u10m_anom  = with_channel(zscore(u10m_xr["u10"]),    "u10m")
vswl1_anom = with_channel(zscore(vswl1_xr["swvl1"]), "vswl1")
vswl2_anom = with_channel(zscore(vswl2_xr["swvl2"]), "vswl2")
z_anom     = with_channel(zscore(z500_xr["z"]),      "z500")

# Combine into one DataArray
#predictors = xr.concat([evv_anom, msl_anom, pr_anom, snsr_anom, tcc_anom, u10m_anom, vswl1_anom, vswl2_anom, z_anom], dim="channel")
predictors = xr.concat([z_anom, tcc_anom, snsr_anom, evv_anom], dim="channel")
X = predictors.values

# Load target datasets
target  = xr.open_dataset("target_var/AFR_1984-2023_daily_tw_heatwaves_day.nc")
y = with_channel(target["tw_heatwaves_day"], "label").values

years = pd.to_datetime(target.time.values).year

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

#model for turning hyper parameter
def turning_model(n_filters=32, dropout_rate=0.3, l2_strength=1e-4, lr=1e-3):
    return unet_regressor(input_shape=(321, 321, 4), n_filters=n_filters, dropout_rate=dropout_rate, l2_strength=l2_strength, lr=lr, loss_fn=loss_fn)

model = KerasClassifier(build_fn=turning_model, epochs=20, batch_size=16, verbose=1)

param_dist = {
    "n_filters": [16, 32, 64],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5],
    "l2_strength": [1e-2, 1e-3, 1e-4, 1e-5],
    "lr": [1e-2, 1e-3, 1e-4],
    "batch_size": [16, 32, 64, 128, 256, 512]
}

def recall_scorer(y_true, y_pred):
    return recall_score(y_true.flatten(), (y_pred.flatten() > 0.5).astype(np.uint8), zero_division=0)

class YearModulo4Split(BaseCrossValidator):
    def __init__(self, years):
        self.years = np.array(years)
    
    def split(self, X, y=None, groups=None):
        for fold in range(4):
            train_idx = np.where((self.years % 4) != fold)[0]
            test_idx = np.where((self.years % 4) == fold)[0]
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 4

years = pd.to_datetime(target.time.values).year
cv = YearModulo4Split(years)

recall_metric = make_scorer(recall_scorer, greater_is_better=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=50,
                                   cv=cv,
                                   verbose=2,
                                   scoring=recall_metric,
                                   n_jobs=1,
                                   loss_fn=loss_fn)

random_search.fit(X, y, callbacks=[early_stop])

best_model = random_search.best_estimator_
best_params = random_search.best_params_


keras_model = best_model.model
keras_model.save("best_unet_model.h5")

with open("best_params.json", "w") as f:
    json.dump(best_params, f)

print(random_search.best_params_, random_search.best_score_)

print("Model and best params saved successfully.")
