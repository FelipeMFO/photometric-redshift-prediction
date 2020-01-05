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
