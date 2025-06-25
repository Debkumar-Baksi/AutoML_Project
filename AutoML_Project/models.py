from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

def get_model(name, params, task_type="classification"):
    if task_type == "classification":
        if name == "xgboost":
            return XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        elif name == "lightgbm":
            return LGBMClassifier(**params, verbose=-1)
        elif name == "catboost":
            return CatBoostClassifier(**params, verbose=0)
        elif name == "logistic":
            return LogisticRegression(**params)
        elif name == "random_forest":
            return RandomForestClassifier(**params)
        elif name == "decision_tree":
            return DecisionTreeClassifier(**params)
        elif name == "svm":
            return SVC(**params)
        elif name == "naive_bayes":
            return GaussianNB(**params)

    else:  # regression
        if name == "xgboost":
            return XGBRegressor(**params)
        elif name == "lightgbm":
            return LGBMRegressor(**params, verbose=-1)
        elif name == "catboost":
            return CatBoostRegressor(**params, verbose=0)
        elif name == "linear":
            return LinearRegression(**params)
        elif name == "ridge":
            return Ridge(**params)
        elif name == "lasso":
            return Lasso(**params)
        elif name == "random_forest":
            return RandomForestRegressor(**params)
        elif name == "decision_tree":
            return DecisionTreeRegressor(**params)
        elif name == "svr":
            return SVR(**params)

