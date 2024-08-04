import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from process_data import generate_dataset
from tqdm import tqdm


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


batch_size = 4096

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1,  # suppress output
    'force_col_wise': True  # remove testing overhead
}

print(X_val.shape[0])
num_batches = int(X_val.shape[0] / batch_size) - 1

online_val_losses = []
# for i in range(1, int(X_val.shape[0] / batch_size) - 1):
for i in tqdm(range(1, num_batches), desc="Training Progress"):
    train_data = lgb.Dataset(X_val[:i*batch_size], label=Y_val[:i*batch_size])

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Make predictions on test data
    Y_pred = model.predict(X_val)

    # Evaluate the model
    mse = mean_squared_error(Y_val, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_val, Y_pred)
    mae = mean_absolute_error(Y_val, Y_pred) 
    online_val_losses.append([mae, rmse, r2])
    

avg_losses = np.mean(online_val_losses, axis=0)
print(f"Average MAE: {avg_losses[0]:.4f}, RMSE: {avg_losses[1]:.4f}, R2: {avg_losses[2]:.4f}")



# test set
num_batches = int(X_test.shape[0] / batch_size) - 1

online_test_losses = []
# for i in range(1, int(X_val.shape[0] / batch_size) - 1):
for i in tqdm(range(1, num_batches), desc="Training Progress"):
    train_data = lgb.Dataset(X_test[:i*batch_size], label=Y_test[:i*batch_size])

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Make predictions on test data
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred) 
    online_test_losses.append([mae, rmse, r2])
    

avg_losses = np.mean(online_test_losses, axis=0)
print(f"Average MAE: {avg_losses[0]:.4f}, RMSE: {avg_losses[1]:.4f}, R2: {avg_losses[2]:.4f}")



