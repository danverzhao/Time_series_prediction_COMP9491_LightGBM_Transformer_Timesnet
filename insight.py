import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from process_data import generate_dataset, generate_grouped_dataset, create_batches, N_STOCKS
from tqdm import tqdm




#===============================
dir = 'data_split_True'

X_train = np.load(f'{dir}/X_train.npy')
Y_train = np.load(f'{dir}/y_train.npy')
# Y_train = Y_train.flatten()

print('train sets:')
print(X_train.shape)
print(Y_train.shape)

X_test = np.load(f'{dir}/X_test.npy') 
Y_test = np.load(f'{dir}/y_test.npy') 
# Y_test = Y_test.flatten()

print('test sets:')
print(X_test.shape)
print(Y_test.shape)

print("X_train stats:")
print("Min:", np.min(X_train))
print("Max:", np.max(X_train))
print("Mean:", np.mean(X_train))
print("NaN count:", np.isnan(X_train).sum())
print("Inf count:", np.isinf(X_train).sum())

print("\nY_train stats:")
print("Min:", np.min(Y_train))
print("Max:", np.max(Y_train))
print("Mean:", np.mean(Y_train))
print("NaN count:", np.isnan(Y_train).sum())
print("Inf count:", np.isinf(Y_train).sum())