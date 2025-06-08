from tabpfn import TabPFNRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

name = 'Nikola Jokic'
season = '2023-24'


df_player = pd.read_csv(f'datasets\{name}.csv')

# Train
# X = df_player.drop(columns=['NEXT_GAME_PTS'])
X = df_player.drop(columns=['NEXT_GAME_PTS_PER_MIN'])

# Test
# Y = df_player.NEXT_GAME_PTS
Y = df_player.NEXT_GAME_PTS_PER_MIN


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

tab_reg = TabPFNRegressor()
cat_reg = CatBoostRegressor(verbose=0)

models = [tab_reg, cat_reg]

# 5 fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model in models:
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        r2_scores.append(r2_score(y_test_fold, y_pred))
        mae_scores.append(mean_absolute_error(y_test_fold, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred)))

    print(type(model).__name__)
    print("R² scores:", r2_scores)
    print("Mean R²:", np.mean(r2_scores))
    print("MAE scores:", mae_scores)
    print("Mean MAE:", np.mean(mae_scores))
    print("RMSE scores:", rmse_scores)
    print("Mean RMSE:", np.mean(rmse_scores))