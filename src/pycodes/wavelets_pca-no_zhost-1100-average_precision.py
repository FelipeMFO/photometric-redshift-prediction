import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.8608619004357136
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=7, min_samples_split=17, n_estimators=100)),
    StackingEstimator(estimator=LogisticRegression(C=1.0, dual=False, penalty="l1")),
    MaxAbsScaler(),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.7500000000000001, n_estimators=100), step=0.1),
    StackingEstimator(estimator=GaussianNB()),
    MinMaxScaler(),
    XGBClassifier(learning_rate=0.1, max_depth=4, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.6000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
