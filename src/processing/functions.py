import os
import json
import numpy as np
import pandas as pd
import pywt as wt
from sklearn.decomposition import PCA
from glob import glob

#-------------------- Commun Functions --------------------

def get_raw_files(path):
    """Get raw files path from data/raw dir."""
    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.DAT' in file:
                files.append(os.path.join(r, file))
    
    return files

def read_sn(file_name):
    """Get dict from file path."""

    def to_number(x):
        try:
            return float(x)
        except:
            return x

    with open(file_name, 'r') as fp:
        lines = fp.readlines()[:-1]
        
    lines = [line.replace('\n','').replace('+-','').split() for line in lines]
    lines = [line[:line.index('Commun#')] if '#' in line else line for line in lines]
    lines = [line for line in lines if line != list()]
    
    meta_data = {line[0].replace(':',''): list(map(to_number, line[1:]))
                 for line in lines if not 'OBS:' in line}
    
    meta_data = {k: v[0] if len(v)==1 else v for k, v in meta_data.items()}
    df_obs = pd.DataFrame([line[1:] for line in lines if 'OBS:' in line], columns=meta_data.pop('VARLIST')).drop(columns=['FIELD'])
    
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

#-------------------- Gen Labels Functions --------------------

def get_label(file_path): #get the filter dictionary from a raw data and its X points to predict
    """Read the file with read_sn and get label and ID from objects."""
    label = read_sn(file_path)['SIM_COMMENT'][3]
    ID = 'SN' + str(read_sn(file_path)['SNID'])
    
    return ID, label

def get_df_label(files):
    """Receives a list of files and returns a DataFrame with ID and Supernova type."""
    df = pd.DataFrame(columns = ['ID', 'type'])
    for i, obj in enumerate(files):
        ID, tp = get_label(obj)
        df.loc[i] = [ID, tp]
    df['type_bool'] = False 
    df['type_bool'].loc[np.where(df['type'] == 'Ia')] = True
    return df

def get_df_redshift(zpsec_df):
    zpsec_df.columns = ['ID', 'REDSHIFT_SPEC']
    zpsec_df['ID'] = 'SN' + zpsec_df['ID'].astype(str)
    return zpsec_df

#-------------------- Gaussian Process Functions --------------------

def get_filter_data(file_path):
    '''Get the filter dictionary from a raw data and its X points to predict.'''
    data = read_sn(file_path)['df']
    X = np.linspace(data.MJD.min(), data.MJD.max(), 100)
    data_dict = {band: df[['MJD', 'FLUXCAL', 'FLUXCALERR']].values for band, df in data.groupby('FLT')}   
    ID = 'SN' + str(read_sn(file_path)['SNID'])
    
    return ID, data_dict, X

def get_df_filter_data(files):
    df = pd.DataFrame(columns = ['ID', 'desg','desi','desr','desz', 'Xaxis'])
    for i, obj in enumerate(files):
        ID, filters, time = get_filter_data(obj)
        df.loc[i] = [ID, filters['desg'], filters['desi'], filters['desr'], filters['desz'], time]
    return df

def gplc(index,x, y, yerr, npoints = 100, n_restarts_optimizer = 100):
    '''Applies Gaussian Process using filter values.'''
    #print(index)
    X = x.reshape(-1, 1)#df.loc[:, x].values.reshape(-1, 1)
    y = y#df.loc[:, y].values
    yerr = yerr#df.loc[:, yerr].values
    
    Xmin, Xmax = X.min(), X.max()
    vary = y.var()
    
    i = y > 0
    
    yerr = yerr[i]/y[i]
    y = np.log(y[i])
    X = X[i]

    const = ConstantKernel(vary, constant_value_bounds=(1e-5, 2*vary))
    rbf = RBF(length_scale = .5*Xmax, length_scale_bounds=(1e-05, Xmax))

    kernel = const * rbf
    
    GPR = GaussianProcessRegressor(kernel = kernel, alpha = yerr**2, n_restarts_optimizer = n_restarts_optimizer)
    
    GPR.fit(X, y)
    
    nX = np.linspace(Xmin, Xmax, npoints).reshape(-1, 1)
    
    ny, covy = GPR.predict(nX, return_cov=True)

    expy = np.exp(ny)
    cov = (expy * expy.reshape(-1, 1)) * covy
    # Xaxis -> time, Yaxis -> result, Y axis error
    return nX.ravel(), expy, np.sqrt(cov.diagonal())

def wavelets(obj, wavelet = 'sym2', mlev = 2 ):
    '''Applies wavelets transform on each filter GP interpolation.'''
    wav = wt.Wavelet(wavelet)
    coeff_g = np.array(wt.swt(obj['desg_GP'][1], wav, level=mlev)).flatten()
    coeff_i = np.array(wt.swt(obj['desi_GP'][1], wav, level=mlev)).flatten()
    coeff_r = np.array(wt.swt(obj['desr_GP'][1], wav, level=mlev)).flatten()
    coeff_z = np.array(wt.swt(obj['desz_GP'][1], wav, level=mlev)).flatten()
    coeffs = np.concatenate([coeff_g,coeff_i,coeff_r,coeff_z])
    #coeffs = coeffs.reshape(-1, 1)
    return coeffs

def get_df_PCA(df_GP_wav, PCA_n = 20, print_variance = True):
    '''Applies PCA and return data frame with features.'''
    #wav = df_GP_wav['wavelets']
    wav = np.array(df_GP_wav['wavelets'].tolist())
    pca = PCA(n_components = PCA_n)
    pca.fit(wav)  
    if print_variance:
        print('Explained variance ratio: ', pca.explained_variance_ratio_)
        print('\n', 'Singular values: ', pca.singular_values_)
    features = pca.fit_transform(wav)

    columns_names = []
    for i in range(PCA_n):
        columns_names.append(f'f{i+1}')

    df_features = pd.DataFrame(features, columns=columns_names)
    return df_features
