import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import StackingEstimator

# Score on the training set was:0.9682044603515342
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=21, p=1, weights="uniform")),
        StackingEstimator(estimator=make_pipeline(
            RBFSampler(gamma=0.1),
            ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=18, min_samples_split=7, n_estimators=100)
        ))
    ),
    MaxAbsScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.9000000000000001, min_samples_leaf=12, min_samples_split=9, n_estimators=100, subsample=0.8500000000000001)
)

