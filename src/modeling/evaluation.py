import os 
import sys
import time
import h2o

import pandas as pd
import numpy as np
from ..modeling.functions_autoML import gen_predictions

def get_delta_z(zphot, zspec):
    return (zphot - zspec)/(1 + zspec)

def get_prediction_bias(delta_z):
    return np.median(delta_z)

def get_MAD(delta_z, median_delta_z):
    return np.median(np.abs(delta_z - median_delta_z))

def get_sigma_MAD(MAD):
    return 1.4826*MAD

def get_n_fraction(delta_z):
    return sum(np.abs(delta_z) > 0.05)/len(delta_z)

def get_metrics_from_batchs(dict_batchs: dict,
                            df: pd.DataFrame) -> dict:
    """[summary]

    Args:
        dict_batchs (dict): dictionary with amls objects as values,
        and their size as keys.
        df (pd.DataFrame): DataFrame with all data (only IA type)

    Returns:
        dict: [description]
    """
    ans = {}
    for key in dict_batchs.keys():
        preds, len_train, len_validation = \
            gen_predictions(df.iloc[-int(key[5:]):], dict_batchs[key].get_best_model())
        df_sigma_mad = df.iloc[-len_validation:]
        df_sigma_mad['zphot'] = preds.as_data_frame()['predict'].tolist()

        delta_z = get_delta_z(df_sigma_mad.zphot.values, df_sigma_mad.REDSHIFT_SPEC.values)
        prediction_bias = get_prediction_bias(delta_z)
        MAD = get_MAD(delta_z, prediction_bias)
        sigma_MAD = get_sigma_MAD(MAD)
        n_fraction = get_n_fraction(delta_z)
        ans[f"prediction_bias_{key}"] = get_prediction_bias(delta_z)
        ans[f"sigma_MAD_{key}"] = sigma_MAD
        ans[f"n_fraction_{key}"] = n_fraction
    
    return ans