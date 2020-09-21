#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 02:14:13 2019

@author: felipematheus
"""

import pandas as pd
import numpy as np

import pywt as wt
import pickle

import pymc3 as pm
import matplotlib.pyplot as plt

def get_xys(df): #get the know values of a specific filter
    y = df[:, 1]
    yerr = df[:, 2]
    x_known = df[:, 0]
    
    return x_known, y, yerr

def cleaning_df(df, method = '', clean_neg = True, percentage = 0.5):
    if clean_neg: #verifies if the value is negative and if it is under the error margin, if it is, turn to zero
        df[(df[:, 1] < 0) & (df[:, 1] > -df[:, 2]) , 1] = 0
        df = df[(df[:, 1] > 0)] #otherwise just cut off
    if method == 'std_dev': #cuts the points with error over the mean error + 1 std
        threshold = df.mean(axis = 0)[2] + df.std(axis = 0)[2]
        df_filter = df[(threshold>df[:,2])]
    elif method == 'percentage':
        threshold = df.max(axis = 0)[1] * percentage
        df_filter = df[(threshold>df[:,2])]
    else:
        df_filter = df
    return df_filter

def get_df(file_path): #get the filter dictionary from a raw data and its X points to predict
    data = read_sn(file_path)['df']
    X = np.linspace(data.MJD.min(), data.MJD.max(), 100)
    data_dict = {band: df[['MJD', 'FLUXCAL', 'FLUXCALERR']].values for band, df in data.groupby('FLT')}   
    
    return data_dict, X

def GP_prep(file, filters, clean_method = '', gp_method = 'm32'):
    
    data_dict, t = get_df(file)
    t = t.reshape(len(t),1)
        
    mus = {filters[0] : [], filters[1] : [], filters[2] : [], filters[3] : []}
    stds = {filters[0] : [], filters[1] : [], filters[2] : [], filters[3] : []}
    

    for band, dat in data_dict.items():
        #with error treatment std_dev
        x, y, yerr = get_xys(cleaning_df(data_dict[band], method = clean_method, clean_neg = True))
        x = x.reshape(len(x),1)
        
        if gp_method == 'm32':
            fit = GP_matern32(t,x,y,yerr)
        elif gp_method == 'm52':
            fit = GP_matern52(t,x,y,yerr)
        elif gp_method == 'exp':
            fit = GP_expquad(t,x,y,yerr)
        elif gp_method == 'rat':
            fit = GP_ratquad(t,x,y,yerr)
    
        stds[band] = fit['mu_total']
        mus[band] = fit['sd_total']
                
    return t,mus,stds
    

def get_wavelets(file, keys, wavelet = 'sym2', mlev = 2):
    wav = wt.Wavelet(wavelet)
    
    xstar, mu, stds = GP_prep(file, keys)
    
    for filt in keys: 
        coeffs = [np.array(wt.swt(mu[filt], wav, level=mlev)).flatten()]

    return np.concatenate(coeffs)


def GP_matern32(t,x,y,yerr):
    np.random.seed(9)
    with pm.Model() as model:
        eta = pm.HalfCauchy("eta", beta=2, testval=2.0)
        lengthscale = pm.Gamma("lenght", alpha=4, beta=0.1)
        cov = 500*eta**2 * pm.gp.cov.Matern32(1, lengthscale)        

        gp = pm.gp.Marginal(cov_func=cov)

        y_ = gp.marginal_likelihood("y", X=x, y=y, noise=yerr)

        #mp = pm.find_MAP(include_transformed=True)

        mu, var = gp.predict(t,  diag=True)
        fit = pd.DataFrame({"t": t.flatten(),
                        "mu_total": mu,
                        "sd_total": np.sqrt(var)})
        return fit

def GP_matern52(t,x,y,yerr):
    np.random.seed(9)
    with pm.Model() as model:
        eta = pm.HalfCauchy("eta", beta=2, testval=2.0)
        lengthscale = pm.Gamma("lenght", alpha=4, beta=0.1)
        cov = 500*eta**2 * pm.gp.cov.Matern52(1, lengthscale)        

        gp = pm.gp.Marginal(cov_func=cov)

        y_ = gp.marginal_likelihood("y", X=x, y=y, noise=yerr)

        #mp = pm.find_MAP(include_transformed=True)

        mu, var = gp.predict(t,  diag=True)
        fit = pd.DataFrame({"t": t.flatten(),
                        "mu_total": mu,
                        "sd_total": np.sqrt(var)})
        return fit
    
def GP_ratquad(t,x,y,yerr):
    np.random.seed(9)
    with pm.Model() as model:
        alfa = 10
        eta = pm.HalfCauchy("N", beta=2, testval=2.0)
        lengthscale = pm.Gamma("L", alpha=4, beta=0.1)
        cov = eta**2 * pm.gp.cov.RatQuad(1, lengthscale, alfa)
    
        gp = pm.gp.Marginal(cov_func=cov)

        y_ = gp.marginal_likelihood("y", X=x, y=y, noise=yerr)

        #mp = pm.find_MAP(include_transformed=True)

        mu, var = gp.predict(t,  diag=True)
        fit = pd.DataFrame({"t": t.flatten(),
                        "mu_total": mu,
                        "sd_total": np.sqrt(var)})
        return fit
    
def GP_expquad(t,x,y,yerr):
    np.random.seed(9)
    with pm.Model() as model:
        eta = pm.HalfCauchy("N", beta=2, testval=2.0)
        lengthscale = pm.Gamma("L", alpha=4, beta=0.1)
        cov = eta**2 * pm.gp.cov.ExpQuad(1, lengthscale)      

        gp = pm.gp.Marginal(cov_func=cov)

        y_ = gp.marginal_likelihood("y", X=x, y=y, noise=yerr)

        #mp = pm.find_MAP(include_transformed=True)

        mu, var = gp.predict(t,  diag=True)
        fit = pd.DataFrame({"t": t.flatten(),
                        "mu_total": mu,
                        "sd_total": np.sqrt(var)})
        return fit
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    import json
import numpy as np
from glob import glob
#from pymultinest.analyse import Analyzer

import multiprocessing as mp

import pandas

from tqdm import tqdm

def to_number(x):
    
    try:
        return float(x)
    
    except:
        
        return x

def read_sn(file_name):

    with open(file_name, 'r') as fp:

        lines = fp.readlines()[:-1]
        
    lines = [line.replace('\n','').replace('+-','').split() for line in lines]
    lines = [line[:line.index('#')] if '#' in line else line for line in lines]
    lines = [line for line in lines if line != list()]
    
    meta_data = {line[0].replace(':',''): list(map(to_number, line[1:]))
                 for line in lines if not 'OBS:' in line}
    
    meta_data = {k: v[0] if len(v)==1 else v for k, v in meta_data.items()}
    
    df_obs = pandas.DataFrame([line[1:] for line in lines if 'OBS:' in line], columns=meta_data.pop('VARLIST')).drop(columns=['FIELD'])
    
    for col in df_obs.drop(columns=['FLT']):
    
        df_obs[col] = df_obs[col].astype(float)
    
    ii = df_obs.FLUXCAL >= 0
    
    df_obs = df_obs[ii]

    df_obs.FLT = (meta_data['SURVEY'] + df_obs.FLT).apply(lambda x: x.lower())

    meta_data['MJD_MIN'] = df_obs.MJD.min()

    df_obs.MJD = df_obs.MJD - meta_data['MJD_MIN']

    cols = df_obs.drop(columns=['FLT']).columns

    meta_data['FILTERS'] = np.unique(df_obs.FLT.values)
        
    meta_data.update({flt: df_obs[df_obs.FLT == flt][['MJD', 'FLUXCAL', 'FLUXCALERR']].values for flt in meta_data['FILTERS']})
        
    meta_data['df'] = df_obs

    meta_data['SNID'] = int(meta_data['SNID'])

    return meta_data

def read_sns(pattern):

    files = glob(pattern)
    
    with mp.Pool(mp.cpu_count()) as p:
        sns = list(tqdm(p.imap(read_sn, files), total=len(files)))
        
    while None in sns:
        
        sns.pop(sns.index(None))
        
    return sns

def paralel_map(func, targets):
    
    cpus = mp.cpu_count()
    
    print("Number of cpus:", cpus)

    with mp.Pool(cpus) as p:

        output = list(tqdm(p.imap(func, targets), total=len(targets)))
        
    return output

    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



    
pickle_files = open("../../models/files.pickle","rb")
files = pickle.load(pickle_files)
filters = ['desg' , 'desi' , 'desr' , 'desz']

