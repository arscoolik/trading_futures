import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from operator import itemgetter

from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def run_algo(X_train, X_test, Y_train, Y_test, algo_name='Random Forest', seed=42, cv_metric='f1', fixed=True, threshold=0.67, cv=None):
    if algo_name == 'Random Forest':
        model = RandomForestClassifier(random_state=seed)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif algo_name == 'XGBoost':
        model = XGBClassifier(random_state=seed, use_label_encoder=False, eval_metric=cv_metric)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=cv_metric, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    Y_pred_is = best_model.predict(X_train)
    Y_pred_is_prob = best_model.predict_proba(X_train)[:, 1]
    if fixed:
        Y_pred_is = (Y_pred_is_prob >= threshold).astype(int)
    Y_pred_oos = best_model.predict(X_test)
    Y_pred_oos_prob = best_model.predict_proba(X_test)[:, 1]
    
    if fixed:
        Y_pred_oos = (Y_pred_oos_prob >= threshold).astype(int)
    cls_report = classification_report(Y_test, Y_pred_oos)
    conf_matrix = confusion_matrix(Y_test, Y_pred_oos)

    return {
        'OOS Prediction': Y_pred_oos,
        'IS Prediction': Y_pred_is,
        'Classification Report': cls_report,
        'Confusion Matrix': conf_matrix,
        'Best Model': best_model
    }


from sklearn.metrics import mean_absolute_percentage_error

currency_pair = 'XRPUSD'
data = pd.read_excel(f'{currency_pair}.xlsx', sheet_name='Day', index_col=0)
data.dropna(inplace=True)
# print((data['Return'] < 0).sum() / len(data))
# data['Return'].hist(bins=len(data))
# plt.show()

bins = [-np.inf, -0.5, 0, 0.15, np.inf]
labels = [-2, -1, 1, 2]

SEED = 1

train_val_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=SEED)

X_train = train_val_data.drop(['Return', 'Return Clf'], axis=1)
Y_train = train_val_data[['Return Clf']]

X_test = test_data.drop(['Return', 'Return Clf'], axis=1)
Y_test = test_data[['Return Clf']]

rf_params = {
    'n_estimators': [*range(3,16,3)],
    'max_depth': [None, 1, 2],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'class_weight': ['balanced'], 
}

model = run_algo(X_train, X_test, Y_train, Y_test, algo_name='Random Forest', seed=SEED, cv_metric='f1', fixed=True, threshold=0.67, cv=TimeSeriesSplit(n_splits=7))

Y_pred_oos, Y_pred_is, cls_report, conf_matrix = itemgetter('OOS Prediction', 'IS Prediction', 'Classification Report', 'Confusion Matrix')(model)
print(cls_report)
sns.heatmap(conf_matrix, annot=True, cmap='GnBu')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
