import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def show_grouped_df_dimensions(grouped_df):
    # Get the grouping keys
    group_keys = list(grouped_df.groups.keys())
    
    # Get the first group
    first_group = next(iter(grouped_df))
    group_name, group_data = first_group
    
    # If there's a second level of grouping, get the first subgroup
    if isinstance(group_data, pd.core.groupby.generic.DataFrameGroupBy):
        subgroup_name, subgroup_data = next(iter(group_data))
    else:
        subgroup_data = group_data

    print(f"Number of top-level groups: {len(group_keys)}")
    print(f"Number of second-level groups (if applicable): {len(group_data) if isinstance(group_data, pd.core.groupby.generic.DataFrameGroupBy) else 'N/A'}")
    print(f"Number of rows in each subgroup: {len(subgroup_data)}")
    print(f"Number of columns: {len(subgroup_data.columns)}")
    print(f"Column names: {', '.join(subgroup_data.columns)}")


def create_windowed_dataset_in(df, window_size):
    # Separate features and target
    columns_to_drop=['target', 'stock_id', 'date_id']
    features = df.drop(columns=columns_to_drop, axis=1).values
    target = df['target'].values

    X_dataset = []
    y_dataset = []

    for i in range(len(df) - window_size + 1):
        X_dataset.append(features[i:i+window_size].flatten())
        y_dataset.append(target[i+window_size-1])

    return np.array(X_dataset), np.array(y_dataset)




def create_windowed_dataset(df, window_size=5, target_column='target', fill_value=0):
    df = df.drop(['time_id', 'row_id', 'seconds_in_bucket'], axis=1)

    # Normalise certain columns (e.g. non category)
    columns_to_normalize = ['imbalance_size','reference_price','matched_size','far_price','near_price','bid_price','bid_size','ask_price','ask_size','wap']

    scaler = StandardScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Fill NaN with 0
    df = df.fillna(0)

    print(f'rows num: {len(df)}')
    nan_count = df['far_price'].isnull().sum()
    print(f'far price NaNs: {nan_count}')
    nan_count = df['near_price'].isnull().sum()
    print(f'near price NaNs: {nan_count}')

    print(df.head())

    #===================================================================
    # group to: days -> stock_ids -> prices for that stock on that day
    # grouped.shape (480, 200, 55)
    grouped_df = df.groupby(['date_id', 'stock_id'])
    print(type(grouped_df))
    show_grouped_df_dimensions(grouped_df)


    X_all = []
    y_all = []
    for group_name, group_data in grouped_df:
        
        X, y = create_windowed_dataset_in(group_data, window_size=5) 
        X_all.append(X)
        y_all.append(y)

    # Concatenate all X and y arrays
    X_concatenated = np.concatenate(X_all, axis=0)
    y_concatenated = np.concatenate(y_all, axis=0)

    return X_concatenated, y_concatenated




df = pd.read_csv('train.csv')
# train
print(len(df))
df = df[:int(len(df) * 0.85)]
print(df.shape)
print(type(df))

X_train, y_train = create_windowed_dataset(df, window_size=5, target_column='target', fill_value=0)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

# test
df = df[int(len(df) * 0.85) + 10:]
print(df.shape)
print(type(df))

X_test, y_test = create_windowed_dataset(df, window_size=5, target_column='target', fill_value=0)
print(f'X_train: {X_test.shape}, y_train: {y_test.shape}')
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

