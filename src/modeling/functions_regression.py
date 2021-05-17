import pandas as pd
import numpy as np


from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#-------------------- Cross Validation Functions --------------------

def divide_validation(X, y, splits):
    """
    Once we need test, train and validation sets, this function divides DF in {splits} splits,
    in order to destinate ~ 1100 to normal 80-20 train and test. Then we Validate with the ~ {len(df) - splits} lines
    """
    kf = KFold(n_splits = splits)
    X_validate = [] # List with Xs to validate
    X_train_test = []  # List with Xs to train and test
    y_validate = [] # List with ys to validate
    y_train_test = []  # List with ys to train and test

    for train_index_real, test_index_real in kf.split(X):
        #print("X_validate:", train_index_real, "X_train_test:", test_index_real)
        X_validate.append(X.iloc[train_index_real])
        X_train_test.append(X.iloc[test_index_real])
        y_validate.append(y[train_index_real])
        y_train_test.append(y[test_index_real])

    return (X_train_test, X_validate, y_train_test, y_validate)

def Kfold(X, y, splits):
    """
    Normal K-Fold
    """
    kf = KFold(n_splits = splits)
    X_train, X_test, y_train, y_test = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train.append(X.iloc[train_index])
        X_test.append(X.iloc[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    return (X_train, X_test, y_train, y_test)

#-------------------- Regression Functions --------------------


def linear_regression(X, y, splits_validation, splits_KFold, regressor):
    """
    Divides DF in splits_validation splits, in order to destinate ~ 1100 to ordinary K-Fold 80-20 train and test.
    Then we Validate with the other rows
    It saves the specific elemnts that will be analysed in a dict format.
    Returns a list of dict with specified keys. Each element corresponds to one K-Fold test inside a splits_validation group. List
    lenght is equal to splits_validation * splits_KFold.
    
    Args:
        X (pandas.core.frame.DataFrame) : Data Frame with colunmns as features and rows as objects (Training data)
        y (numpy.ndarray) : Target values 
        splits_validation (int) : Number of Validation splits, it will divide entire data in splits_validation (it should be int(len(df)/1100))
        splits_KFold (int) : Number of K-Folds splits
        regressor (xgboost.sklearn.XGBRegressor) : Regressor parameter from XGBoost
        
    Returns:
        list: return list of dict with specified keys. Each element corresponds to one K-Fold test inside a splits_validation group.
        
    """
    ans = []
    X_train_test, X_validate, y_train_test, y_validate = divide_validation(X, y, splits_validation)
    for i in range(len(X_validate)):
        X_train, X_test, y_train, y_test = Kfold(X_train_test[i], y_train_test[i], splits_KFold)
        for j in range(len(X_train)):
            element = dict.fromkeys(['preds_arr','rmse_arr','mae_arr','r2_arr','matrices_arr','true_preds','true_rms','true_mae','true_r2','models'])
            
            regressor.fit(X_train[j],y_train[j]) #Overwrites last data, do not do Incremental learning
            #print("Fitting " + str(j+1) + 'th model from ' + str(i+1) + 'th validation' )
            element['preds_arr'] = regressor.predict(X_test[j])
            element['rmse_arr'] = np.sqrt(mean_squared_error(y_test[j], element['preds_arr']))
            element['mae_arr'] = mean_absolute_error(y_test[j], element['preds_arr'])
            element['r2_arr'] = r2_score(y_test[j], element['preds_arr'])
            element['matrices_arr'] = np.round((abs(y_test[j]-element['preds_arr'])/y_test[j])*100)
            
            element['true_preds'] = regressor.predict(X_validate[i])
            element['true_rms'] = np.sqrt(mean_squared_error(y_validate[i], element['true_preds'])) # RMES -> Aumenta com maiores variancia
            element['true_mae'] = mean_absolute_error(y_validate[i], element['true_preds'])          # MAE -> Nao varia com variancia
            element['true_r2'] = r2_score(y_validate[i], element['true_preds'])
            element['models'] = regressor
            ans.append(element)         
    return ans

def normal_linear_regression(X, y, splits_KFold, regressor):
    """
    Normal XGBoost linear regressions, dividing the entire Data Set as 80/20 train/test K-Fold. Without Validation.
    It saves the specific elemtns that will be analysed in a dict format.
    
    Args:
        X (pandas.core.frame.DataFrame) : Data Frame with colunmns as features and rows as objects (Training data)
        y (numpy.ndarray) : Target values 
        splits_KFold (int) : Number of K-Folds splits
        regressor (xgboost.sklearn.XGBRegressor) : Regressor parameter from XGBoost
        
    Returns:
        list: return list of dict with specified keys. Each element corresponds to one K-Fold test.    
    """
    ans = []
    X_train, X_test, y_train, y_test = Kfold(X, y, splits_KFold)
    for j in range(len(X_train)):
        element = dict.fromkeys(['preds_arr','rmse_arr','mae_arr','r2_arr','matrices_arr','models'])
        regressor.fit(X_train[j],y_train[j]) #Overwrites last data, do not do Incremental learning
        #print("Fitting " + str(j+1) + 'th model from ' + str(i+1) + 'th validation' )
        element['preds_arr'] = regressor.predict(X_test[j])
        element['rmse_arr'] = np.sqrt(mean_squared_error(y_test[j], element['preds_arr']))
        element['mae_arr'] = mean_absolute_error(y_test[j], element['preds_arr'])
        element['r2_arr'] = r2_score(y_test[j], element['preds_arr'])
        element['matrices_arr'] = np.round((abs(y_test[j]-element['preds_arr'])/y_test[j])*100)

        element['models'] = regressor
        ans.append(element)         
    return ans

#-------------------- Evaluation Functions --------------------


def evaluate(ans_real_19, ans_real_2, ans_normal):
    """
    Print Evaluation based on RMSE, MSE and R2 scores considering validation splits as 18/19, 1/2 and no validation set. 
    """
    list_and_mean = lambda ans, key : (np.array([i[key] for i in ans]) , np.mean(np.array([i[key] for i in ans])))
    l_19, mean_19 = list_and_mean(ans_real_19, 'true_rms')
    l_2, mean_2 = list_and_mean(ans_real_2, 'true_rms')
    l, mean = list_and_mean(ans_normal, 'rmse_arr')
    print(" RMSE Means : \n", 
         "\n Validate 19: ", mean_19,
         "\n Validate 2: ", mean_2,
         "\n No Validation: ", mean)
    l_19, mean_19 = list_and_mean(ans_real_19, 'true_mae')
    l_2, mean_2 = list_and_mean(ans_real_2, 'true_mae')
    l, mean = list_and_mean(ans_normal, 'mae_arr')
    print("\n MAE Means : \n", 
         "\n Validate 19: ", mean_19,
         "\n Validate 2: ", mean_2,
         "\n No Validation: ", mean)
    l_19, mean_19 = list_and_mean(ans_real_19, 'true_r2')
    l_2, mean_2 = list_and_mean(ans_real_2, 'true_r2')
    l, mean = list_and_mean(ans_normal, 'r2_arr')
    print("\n R2 Means : \n", 
         "\n Validate 19: ", mean_19,
         "\n Validate 2: ", mean_2,
         "\n No Validation: ", mean)

#-------------------- Scaling Functions --------------------

def apply_scalers(df, PCA_n):
    """
    Receives DataFrame and number of PCA's features then robust and MinMax scale the features.
    """
    cols = []
    for i in range(PCA_n):
        cols.append(f'f{i+1}')
    X = df.loc[:,cols[0]:cols[-1]]
    #Robust Scaler
    scaler = preprocessing.RobustScaler()
    df_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X, index = df.index, columns=cols)
    #MinMax Scaler
    scaler_mM = preprocessing.MinMaxScaler()
    df_scaled_twice = scaler_mM.fit_transform(df_scaled)
    df_scaled_twice = pd.DataFrame(df_scaled_twice, index = df.index, columns = cols)

    return df_scaled, df_scaled_twice