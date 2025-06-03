from tabpfn import TabPFNRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

name = 'Nikola Jokic'
season = '2023-24'

df_player = pd.read_csv(f'datasets\{name}_{season}.csv')
# print(df_player.columns.to_list())

# Train
X = df_player.drop(columns=['NEXT_GAME_PTS'])
# Test
Y = df_player.NEXT_GAME_PTS

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# print(X_test.PTS)
# print(y_test.columns)
# print(len(X_train))
# print(len(df_player))

reg = TabPFNRegressor()
reg.fit(X_train, y_train)

# Predict a point estimate (using the mean)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

print(y_test, predictions)
