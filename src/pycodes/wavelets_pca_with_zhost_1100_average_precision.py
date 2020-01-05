import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator

# Score on the training set was:0.8625723585503717
exported_pipeline = make_pipeline(
    make_union(
        Nystroem(gamma=0.15000000000000002, kernel="linear", n_components=10),
        StackingEstimator(estimator=make_pipeline(
            StackingEstimator(estimator=LogisticRegression(C=25.0, dual=False, penalty="l1")),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            LogisticRegression(C=0.001, dual=False, penalty="l2")
        ))
    ),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_features=0.8, min_samples_leaf=4, min_samples_split=10, n_estimators=100, subsample=0.9000000000000001)
)

