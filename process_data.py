#!/usr/bin/env python3

# scale & generate features here
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tqdm import tqdm

BUFFER_SIZE = 10000
N_STOCKS = 200
N_DATES = 481
N_SECONDS = 55
TEST_SIZE = 0.1
VAL_SIZE = 0.1
TRAIN_DAY = int(N_DATES * (1 - TEST_SIZE - VAL_SIZE))
VAL_DAY = TRAIN_DAY
TEST_DAY = int(N_DATES * VAL_SIZE + VAL_DAY)


def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float32)

    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f"Decreased by {decrease:.2f}%")

    return df


def build_target(df):
    all_stock_ids = range(N_STOCKS)
    all_date_ids = df["date_id"].unique()
    all_seconds = [i * 10 for i in range(N_SECONDS)]

    multi_index = pd.MultiIndex.from_product(
        [all_stock_ids, all_date_ids, all_seconds],
        names=["stock_id", "date_id", "seconds_in_bucket"],
    )
    df_full = df.set_index(["stock_id", "date_id", "seconds_in_bucket"]).reindex(
        multi_index
    )
    df_full = df_full.fillna(0)
    df_full = df_full.reset_index()

    df_pivoted = df_full.pivot_table(
        values="target", index=["date_id", "seconds_in_bucket"], columns="stock_id"
    )

    df_pivoted = df_pivoted.reset_index(drop=True)
    df_pivoted.columns.name = None

    return df_pivoted


def build_one_pop_target(df):
    n = df.shape[0]
    y = np.zeros((n, N_STOCKS))
    for i, j, t in zip(np.arange(n), df["stock_id"], df["target"]):
        y[i, j] = t
    return y


def generate_df_with_target(path):
    df = pd.read_csv(path)
    df = reduce_mem_usage(df, verbose=1)

    all_stock_ids = range(N_STOCKS)
    all_date_ids = range(N_DATES)
    all_seconds = [i * 10 for i in range(N_SECONDS)]

    multi_index = pd.MultiIndex.from_product(
        [all_stock_ids, all_date_ids, all_seconds],
        names=["stock_id", "date_id", "seconds_in_bucket"],
    )

    df_full = df.set_index(["stock_id", "date_id", "seconds_in_bucket"]).reindex(
        multi_index
    )
    df_full = df_full.fillna(0)
    df_full = df_full.reset_index()

    assert df_full.shape[0] == N_STOCKS * N_DATES * N_SECONDS

    df_full.drop(["time_id", "row_id"], axis=1, inplace=True)

    df_reb = (
        df_full.set_index(["date_id", "seconds_in_bucket"], append=True)
        .swaplevel(1, 0)
        .sort_index(level=2)
        .reset_index()
        .drop(["level_1"], axis=1)
    )

    calc_index_X = lambda x: N_STOCKS * x * N_SECONDS

    df_train = df_reb.iloc[: calc_index_X(TRAIN_DAY)]
    df_val = df_reb.iloc[calc_index_X(VAL_DAY) : calc_index_X(TEST_DAY)]
    df_test = df_reb.iloc[calc_index_X(TEST_DAY) :]

    X_train = df_train.drop(["target"], axis=1)
    y_train = df_train["target"]

    X_val = df_val.drop(["target"], axis=1)
    y_val = df_val["target"]

    X_test = df_test.drop(["target"], axis=1)
    y_test = df_test["target"]

    scaler = StandardScaler()

    df_train = scaler.fit_transform(X_train)
    df_val = scaler.transform(X_val)
    df_test = scaler.transform(X_test)

    # return df_train, df_val, df_test
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def generate_dataset(path, split=True, combine=False, scale=True):
    df = pd.read_csv(path)
    df = reduce_mem_usage(df, verbose=1)

    all_stock_ids = range(N_STOCKS)
    all_date_ids = range(N_DATES)
    all_seconds = [i * 10 for i in range(N_SECONDS)]

    multi_index = pd.MultiIndex.from_product(
        [all_stock_ids, all_date_ids, all_seconds],
        names=["stock_id", "date_id", "seconds_in_bucket"],
    )

    df_full = df.set_index(["stock_id", "date_id", "seconds_in_bucket"]).reindex(
        multi_index
    )
    df_full = df_full.fillna(0)
    df_full = df_full.reset_index()

    assert df_full.shape[0] == N_STOCKS * N_DATES * N_SECONDS

    df_full.drop(["time_id", "row_id"], axis=1, inplace=True)

    X = df_full.drop(["target"], axis=1)
    # y = build_target(df_full)
    y = build_one_pop_target(df_full)

    if combine:
        X = df_full
        X.insert(X.shape[1] - 1, "target", X.pop("target"))

    X = (
        X.set_index(["date_id", "seconds_in_bucket"], append=True)
        .swaplevel(1, 0)
        .sort_index(level=2)
        .reset_index()
        .drop(["level_1"], axis=1)
    )

    calc_index_X = lambda x: N_STOCKS * x * N_SECONDS
    calc_index_y = lambda y: y * N_SECONDS * N_STOCKS

    X_train = X.iloc[: calc_index_X(TRAIN_DAY)]
    y_train = y[: calc_index_y(TRAIN_DAY)]

    X_val = X.iloc[calc_index_X(VAL_DAY) : calc_index_X(TEST_DAY)]
    y_val = y[calc_index_y(VAL_DAY) : calc_index_y(TEST_DAY)]

    X_test = X.iloc[calc_index_X(TEST_DAY) :]
    y_test = y[calc_index_y(TEST_DAY) :]
    print(X_train.head())
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    ct = ColumnTransformer(
        [
            (
                "Scaler",
                StandardScaler(),
                [
                    "imbalance_size",
                    "reference_price",
                    "matched_size",
                    "far_price",
                    "near_price",
                    "bid_price",
                    "bid_size",
                    "ask_price",
                    "ask_size",
                    "wap",
                ],
            )
        ],
        remainder="passthrough",
    )

    X_train_scaled = X_train
    if scale:
        X_train_scaled = ct.fit_transform(X_train)
        X_val_scaled = ct.transform(X_val)
        X_test_scaled = ct.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    if split:
        X_train = np.asarray(np.array_split(X_train_scaled, TRAIN_DAY * N_SECONDS))
        X_val = np.asarray(np.array_split(X_val_scaled, (TEST_DAY - VAL_DAY) * N_SECONDS))
        X_test = np.asarray(np.array_split(X_test_scaled, (N_DATES - TEST_DAY) * N_SECONDS))

        y_train = np.array(np.array_split(y_train, TRAIN_DAY * N_SECONDS))
        y_val = np.array(np.array_split(y_val, (TEST_DAY - VAL_DAY) * N_SECONDS))
        y_test = np.array(np.array_split(y_test, (N_DATES - TEST_DAY) * N_SECONDS))
    else:
        X_train = X_train_scaled
        X_val = X_val_scaled
        X_test = X_test_scaled
    
    
    if scale:
        X_train_scaled = ct.fit_transform(X_train)

    if split == True:
        X_train = np.asarray(np.array_split(X_train_scaled, TRAIN_DAY * N_SECONDS))
        X_val = np.asarray(
            np.array_split(ct.transform(X_val), (TEST_DAY - VAL_DAY) * N_SECONDS)
        )
        X_test = np.asarray(
            np.array_split(ct.transform(X_test), (N_DATES - TEST_DAY) * N_SECONDS)
        )

        y_train = np.array(np.array_split(y_train, TRAIN_DAY * N_SECONDS))
        y_val = np.array(np.array_split(y_val, (TEST_DAY - VAL_DAY) * N_SECONDS))
        y_test = np.array(np.array_split(y_test, (N_DATES - TEST_DAY) * N_SECONDS))



    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def generate_grouped_dataset(path):
    df = pd.read_csv(path)
    df = reduce_mem_usage(df, verbose=1)

    all_stock_ids = range(N_STOCKS)
    all_date_ids = range(N_DATES)
    all_seconds = [i * 10 for i in range(N_SECONDS)]

    multi_index = pd.MultiIndex.from_product(
        [all_stock_ids, all_date_ids, all_seconds],
        names=["stock_id", "date_id", "seconds_in_bucket"],
    )

    df_full = df.set_index(["stock_id", "date_id", "seconds_in_bucket"]).reindex(
        multi_index
    )
    df_full = df_full.fillna(0)
    df_full = df_full.reset_index()

    assert df_full.shape[0] == N_STOCKS * N_DATES * N_SECONDS

    df_full.drop(["time_id", "row_id"], axis=1, inplace=True)
    df_full["imbalance_buy_sell_flag"] += 1
    df_full["seconds_in_bucket"] /= 10

    X = df_full

    groups = []
    X = X.groupby("stock_id")
    stock_df = X.get_group(0)
    for i in range(N_STOCKS):
        stock_df = lag_function(X.get_group(i), ["target"], [1, 2, 3])

        calc_idx = lambda x: x * N_SECONDS

        medians = stock_df.iloc[: calc_idx(TRAIN_DAY)].median()
        stock_df.fillna(medians, inplace=True)
        stock_df.insert(
            stock_df.shape[1] - 1,
            "target",
            stock_df.pop("target"),
        )
        stock_df.insert(
            stock_df.shape[1] - 2,
            "seconds_in_bucket",
            stock_df.pop("seconds_in_bucket"),
        )

        ct = ColumnTransformer(
            [
                (
                    "Scaler",
                    StandardScaler(),
                    [
                        "imbalance_size",
                        "reference_price",
                        "matched_size",
                        "far_price",
                        "near_price",
                        "bid_price",
                        "bid_size",
                        "ask_price",
                        "ask_size",
                        "wap",
                    ],
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        X_train = ct.fit_transform(stock_df.iloc[: calc_idx(TRAIN_DAY)])
        X_val = ct.transform(stock_df.iloc[calc_idx(VAL_DAY) : calc_idx(TEST_DAY)])
        X_test = ct.transform(stock_df.iloc[calc_idx(TEST_DAY) :])

        groups.append((X_train, X_val, X_test))

    # l_transformers = list(ct._iter(fitted=True))
    # cols = []
    # for name, _, elems, _ in l_transformers:
    #    if name == "Scaler":
    #        cols.extend(elems)
    #    else:
    #        cols.extend(stock_df.columns[elems])
    # print(cols)

    return groups


def windowed_dataset(dataset, shuffle=True, n_lags=5, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(n_lags + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_lags + 1))
    dataset = dataset.map(
        lambda window: (
            tf.reshape(window[:-1], (n_lags * N_STOCKS, 15)),
            tf.reshape(window[-1, :, -1], (N_STOCKS,)),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def windowed_grouped_dataset(
    dataset, shuffle=True, n_lags=5, batch_size=32, lagged=False
):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(n_lags + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_lags + 1))
    if lagged:
        dataset = dataset.map(lambda window: (window[:-1, :-1], window[-1, -1]))
    else:
        dataset = dataset.map(lambda window: (window[:, :-1], window[-1, -1]))
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def lag_function(df, columns_to_lag, numbers_of_days_to_lag):

    df_indexed = df.set_index(["stock_id", "seconds_in_bucket", "date_id"])

    for column_to_lag in columns_to_lag:
        for number_days_to_lag in numbers_of_days_to_lag:
            df_indexed[f"lag{number_days_to_lag}_{column_to_lag}"] = df_indexed.groupby(
                level=["stock_id", "seconds_in_bucket"]
            )[column_to_lag].shift(number_days_to_lag)

    df_indexed.reset_index(inplace=True)

    return df_indexed


def precompute_sequences(stock_data, window_size):
    # Convert DataFrame columns to NumPy arrays
    num_features = stock_data.columns[
        (stock_data.columns != "seconds_in_bucket") & (stock_data.columns != "target")
    ]
    stock_data_num = stock_data[num_features].values
    stock_data_cat = stock_data[["seconds_in_bucket"]].values

    # Pre-compute all sequences
    all_sequences_num = [
        stock_data_num[max(0, i - window_size + 1) : i + 1]
        for i in range(len(stock_data))
    ]
    all_sequences_cat = [
        stock_data_cat[max(0, i - window_size + 1) : i + 1]
        for i in range(len(stock_data))
    ]

    # Add padding if necessary
    padded_sequences_num = [
        np.pad(seq, ((window_size - len(seq), 0), (0, 0)), "constant")
        for seq in all_sequences_num
    ]
    padded_sequences_cat = [
        np.pad(seq, ((window_size - len(seq), 0), (0, 0)), "constant")
        for seq in all_sequences_cat
    ]

    # Combine numerical and categorical features
    combined_sequences = np.array(
        [
            np.concatenate([num, cat], axis=-1)
            for num, cat in zip(padded_sequences_num, padded_sequences_cat)
        ]
    )

    # Extract targets
    targets = stock_data["target"].values

    return combined_sequences, targets


def create_batches(
    data,
    window_size,
    max_time_steps=55,
):

    print(f"Creating batches...")
    grouped = data.groupby(["stock_id", "date_id"])
    all_batches = []
    all_targets = []

    def get_sequence(precomputed_data, time_step):
        combined_sequences, targets = precomputed_data
        return combined_sequences[time_step], targets[time_step]

    for _, group in tqdm(grouped, desc="Processing groups"):
        # Precompute sequences for the current group
        precomputed_data = precompute_sequences(group, window_size)

        # Initialize containers for group sequences and targets
        group_sequences = []
        group_targets = []

        # Iterate over the time steps and retrieve precomputed sequences
        for time_step in range(max_time_steps):
            sequence, target = get_sequence(precomputed_data, time_step)
            if sequence.size > 0:
                group_sequences.append(sequence)
                group_targets.append(target)

        # Extend the main batches with the group's sequences and targets
        all_batches.extend(group_sequences)
        all_targets.extend(group_targets)

    return np.asarray(all_batches), np.asarray(all_targets)
