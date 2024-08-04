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
warnings.filterwarnings('ignore')
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
    'feature_fraction': 0.9,
    'verbose': -1,  # suppress output
    'force_col_wise': True  # remove testing overhead
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


#================================

#online




# Function for evaluation
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2

# Evaluate initial model
print("Initial Model Performance:")
mae, rmse, r2 = evaluate_model(model, X_test, Y_test)
print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")






def online_learning_with_pruning(model, X, y, X_val, y_val, num_rounds=1, max_trees=100):
    # Create a LightGBM dataset from the new data
    new_data = lgb.Dataset(X, label=y)
    
    # Create a validation dataset
    val_data = lgb.Dataset(X_val, label=y_val)
    
    # Get the current number of trees
    current_trees = model.num_trees()
    
    # Update parameters for continued training
    update_params = model.params.copy()
    update_params['num_trees'] = max_trees  # Set the maximum number of trees
    
    if current_trees >= max_trees:
        # If we're at or above max trees, train a small number of new trees
        trees_to_add = min(num_rounds, max_trees // 10)  # Add up to 10% new trees
        update_params['num_trees'] = current_trees + trees_to_add
    else:
        # If we're below max trees, we can add more
        trees_to_add = max_trees - current_trees
    
    # Continue training the model
    updated_model = lgb.train(
        params=update_params,
        train_set=new_data,
        num_boost_round=trees_to_add,
        init_model=model,
        valid_sets=[val_data],
        early_stopping_rounds=5,
        verbose_eval=False
    )
    
    # Prune the model if it exceeds max_trees
    if updated_model.num_trees() > max_trees:
        updated_model = lgb.create_tree_learner(updated_model.model_to_string(), max_trees)
    
    return updated_model




# Parameters
ONLINE_EPOCHS = 4
BATCH_SIZE = 1024
MAX_TREES = 150  

# Online learning on validation set with pruning
online_val_losses = []
for i in tqdm(range(0, len(X_val), BATCH_SIZE), desc="Online Learning on Validation Set"):
    X_batch = X_val[i:i+BATCH_SIZE]
    y_batch = Y_val[i:i+BATCH_SIZE]
    
    # Evaluate before update
    mae, rmse, r2 = evaluate_model(model, X_batch, y_batch)
    online_val_losses.append([mae, rmse, r2])
    
    # Update model with pruning
    model = online_learning_with_pruning(model, X_batch, y_batch, X_val, Y_val, 
                                         num_rounds=ONLINE_EPOCHS, max_trees=MAX_TREES)
    
    print(f"Current number of trees: {model.num_trees()}")

print("Online Learning Performance on Validation Set:")
avg_losses = np.mean(online_val_losses, axis=0)
print(f"Average MAE: {avg_losses[0]:.4f}, RMSE: {avg_losses[1]:.4f}, R2: {avg_losses[2]:.4f}")

# Online learning on test set with pruning
online_test_losses = []
for i in tqdm(range(0, len(X_test), BATCH_SIZE), desc="Online Learning on Test Set"):
    X_batch = X_test[i:i+BATCH_SIZE]
    y_batch = Y_test[i:i+BATCH_SIZE]
    
    # Evaluate before update
    mae, rmse, r2 = evaluate_model(model, X_batch, y_batch)
    online_test_losses.append([mae, rmse, r2])
    
    # Update model with pruning
    model = online_learning_with_pruning(model, X_batch, y_batch, X_val, Y_val, 
                                         num_rounds=ONLINE_EPOCHS, max_trees=MAX_TREES)
    
    print(f"Current number of trees: {model.num_trees()}")

print("Online Learning Performance on Test Set:")
avg_losses = np.mean(online_test_losses, axis=0)
print(f"Average MAE: {avg_losses[0]:.4f}, RMSE: {avg_losses[1]:.4f}, R2: {avg_losses[2]:.4f}")

# Final evaluation on entire test set
print("Final Model Performance on Entire Test Set:")
mae, rmse, r2 = evaluate_model(model, X_test, Y_test)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
print(f"Final number of trees: {model.num_trees()}")