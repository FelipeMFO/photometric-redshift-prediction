import pandas as pd
import numpy as np
import h2o

from h2o.automl import H2OAutoML

def auto_ML(df, n_models, validation_ratio = .5):
    """
    Initialize h2o, a new Auto ML object and already train it applying verification following the validation ratio.
    """
    h2o.init(ip="localhost", port=54323)
    
    X = df.loc[:,'f1':'REDSHIFT_SPEC']
    y = df.loc[:,'REDSHIFT_SPEC']
    aml = H2OAutoML(max_models=n_models, seed=1)
    
    train, test = h2o.H2OFrame(X).split_frame(ratios=[validation_ratio])
 
    aml.train(x=list(df.loc[:,'f1':'f20'].columns),
          y='REDSHIFT_SPEC',
          training_frame=train,
          leaderboard_frame=test)

    lb = aml.leaderboard
    print("Leaderboard: ", lb.head(rows=lb.nrows), '\n')
    print("Leader: ", aml.leader, "\n")
    
    return aml

def save_aml_models(aml):
    """
    Save all aml object models on the respective folder.
    """
    aml_pd = aml.leaderboard.as_data_frame(use_pandas=True)
    lb = aml.leaderboard
    path = f"../../models/mojo_{len(lb)-2}_ensemble/"

    for i in range(len(lb)):
        model = h2o.get_model(aml_pd['model_id'][i])
        model.save_mojo(path, force = True)
        print(f"Model {aml_pd['model_id'][i]} saved.")

def gen_predictions(df, model, ratio = .5, return_len = True):
    """
    Receives DataFrame and model, it split the frame and returns predictions and len of splits.
    """
    X = df.loc[:,'f1':'REDSHIFT_SPEC']
    y = df.loc[:,'REDSHIFT_SPEC']
    train, test = h2o.H2OFrame(X).split_frame(ratios=[ratio], seed=1)
    preds = model.predict(h2o.H2OFrame(X.iloc[-len(test):].loc[:,'f1':'f20']))
    print('\n', f"{len(train)} train/test objects ",
          "\n",
         f"and {len(test)} validation objects")
    if return_len:
        return preds, len(train), len(test)
    return preds