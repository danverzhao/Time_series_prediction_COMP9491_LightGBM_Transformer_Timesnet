import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from process_data import generate_dataset


# Suppress warnings
# warnings.filterwarnings('ignore')
dir = 'grouped_data/5_days'

X_train = np.load(f'{dir}/X_train.npy')
Y_train = np.load(f'{dir}/y_train.npy')
# Y_train = Y_train.flatten()
X_train = X_train.reshape(len(X_train), -1)

print('train sets:')
print(X_train.shape)
print(Y_train.shape)

X_test = np.load(f'{dir}/X_test.npy') 
Y_test = np.load(f'{dir}/y_test.npy') 
# Y_test = Y_test.flatten()
X_test = X_test.reshape(len(X_test), -1)

print('test sets:')
print(X_test.shape)
print(Y_test.shape)


X_val = np.load(f'{dir}/X_val.npy') 
Y_val = np.load(f'{dir}/y_val.npy') 
# Y_test = Y_test.flatten()
X_val = X_val.reshape(len(X_val), -1)

print('val sets:')
print(X_val.shape)
print(Y_val.shape)



#=============================================================

train_data = lgb.Dataset(X_train, label=Y_train)

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=100)

# Make predictions on test data
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred) 

print('Test')
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Feature importance
feature_importance = model.feature_importance()
feature_names = model.feature_name()

# for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
#     print(f"{name}: {importance}")

# Make predictions on test data
Y_pred = model.predict(X_train)

# Evaluate the model
mse = mean_squared_error(Y_train, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_train, Y_pred)
mae = mean_absolute_error(Y_train, Y_pred) 

print('Train')
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")


# Make predictions on test data
Y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(Y_val, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_val, Y_pred)
mae = mean_absolute_error(Y_val, Y_pred) 

print('Val')
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")