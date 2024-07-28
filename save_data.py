import numpy as np
import pandas as pd
from process_data import generate_df_with_target, generate_dataset, generate_grouped_dataset, N_STOCKS, create_batches, windowed_grouped_dataset
from tqdm import tqdm

N_LAGS = 5
groups = generate_grouped_dataset("./train.csv")
X_train, X_val, X_test = groups[0]

for i in tqdm(range(1, N_STOCKS), "Concatenating dataframes"):
    X_train = pd.concat((X_train, groups[i][0]))
    X_val = pd.concat((X_val, groups[i][1]))
    X_test = pd.concat((X_test, groups[i][2]))

X_train, y_train = create_batches(X_train, window_size=N_LAGS)
X_val, y_val = create_batches(X_val, window_size=N_LAGS)
X_test, y_test = create_batches(X_test, window_size=N_LAGS)

dir = 'grouped_data/5_days'

np.save(f'{dir}/X_train.npy', X_train)

# Save y_train
np.save(f'{dir}/y_train.npy', y_train)

# Save X_val
np.save(f'{dir}/X_val.npy', X_val)

# Save y_val
np.save(f'{dir}/y_val.npy', y_val)

# Save X_test
np.save(f'{dir}/X_test.npy', X_test)

# Save y_test
np.save(f'{dir}/y_test.npy', y_test)