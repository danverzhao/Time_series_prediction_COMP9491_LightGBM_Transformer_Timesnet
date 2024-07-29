#!/usr/bin/env python3
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
from process_data import (
    create_batches,
    generate_dataset,
    generate_grouped_dataset,
    windowed_grouped_dataset,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



N_LAGS = 5
N_LAGS_EMB = N_LAGS
BATCH_SIZE = 2**12
EPOCHS = 10
PATIENCE = 20
LEARNING_RATE = 1e-3

N_STOCKS = 200
N_DATES = 481
N_SECONDS = 55


class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, src):
        src = self.embedding(src)
        src = src.unsqueeze(1)  # Add sequence dimension of length 1
        output = self.transformer_encoder(src)
        output = output.squeeze(1)  # Remove sequence dimension
        output = self.dropout(self.relu(self.fc(output)))
        output = self.fc2(output)
        return output.squeeze(-1)  # Remove last dimension to match target shape

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)
    


# generate data
# groups = generate_grouped_dataset("./train.csv")
# X_train, X_val, X_test = groups[0]

# for i in tqdm(range(1, N_STOCKS), "Concatenating dataframes"):
#     X_train = pd.concat((X_train, groups[i][0]))
#     X_val = pd.concat((X_val, groups[i][1]))
#     X_test = pd.concat((X_test, groups[i][2]))


# X_train, y_train = create_batches(X_train, window_size=N_LAGS)
# X_val, y_val = create_batches(X_val, window_size=N_LAGS)
# X_test, y_test = create_batches(X_test, window_size=N_LAGS)

dir = 'grouped_data/5_days'

X_train = np.load(f'{dir}/X_train.npy')
Y_train = np.load(f'{dir}/y_train.npy')
X_train = X_train.reshape(len(X_train), -1)
Y_train = Y_train.flatten()

print('train sets:')
print(X_train.shape)
print(Y_train.shape)

X_test = np.load(f'{dir}/X_test.npy') 
Y_test = np.load(f'{dir}/y_test.npy') 
X_test = X_test.reshape(len(X_test), -1)
Y_test = Y_test.flatten()

print('test sets:')
print(X_test.shape)
print(Y_test.shape)

X_val = np.load(f'{dir}/X_val.npy') 
Y_val = np.load(f'{dir}/y_val.npy') 
X_val = X_val.reshape(len(X_val), -1)
Y_val = Y_val.flatten()

print('val sets:')
print(X_val.shape)
print(Y_val.shape)






# Hyperparameters
batch_size = 1024 
learning_rate = 0.0001
weight_decay = 1e-5  
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 128
validation_split = 0.2


# load model for evaluation
input_dim = X_train.shape[1]  
clf = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)
clf.load_state_dict(torch.load('transformer_models/model2/best_model.pth', map_location=device))

X_train = torch.FloatTensor(X_train).to(device)
Y_train = torch.FloatTensor(Y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
Y_test = torch.FloatTensor(Y_test).to(device)
X_val = torch.FloatTensor(X_val).to(device)
Y_val = torch.FloatTensor(Y_val).to(device)


dataset = TensorDataset(X_train, Y_train)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)



clf = clf.cpu()

# print(f"{X_train.shape}")
train_data = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE)
val_data = DataLoader(TensorDataset(X_val, Y_val), batch_size=BATCH_SIZE)

mae_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
rmse_loss = lambda y_hat, y: torch.sqrt(mse_loss(y_hat, y))

# set model to evaluation mode
clf.eval()






train_losses = []
for X, y in tqdm(train_data, "Evaluating training loss"):
    X = X.to(device)
    y_hat = clf(X).detach().numpy()
    y = y.numpy()
    train_losses.append([
        np.mean(np.abs(y_hat - y)),
        np.sqrt(np.mean((y_hat - y)**2)),
        r2_score(y, y_hat)
    ])
train_losses = np.mean(np.nan_to_num(train_losses, neginf=0), axis=0)

print(train_losses)





val_losses = []
for X_v, y_v in tqdm(val_data, "Evaluating validation loss"):
    X_v = X_v.to(device)
    y_hat = clf(X_v).detach().numpy()
    y_v = y_v.numpy()
    val_losses.append([
        np.mean(np.abs(y_hat - y_v)),
        np.sqrt(np.mean((y_hat - y_v)**2)),
        r2_score(y_v, y_hat)
    ])

val_losses = np.mean(np.nan_to_num(val_losses, neginf=0), axis=0)
print(val_losses)




test_data = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE)
test_losses = []
for X, y in tqdm(test_data, "Evaluating test loss"):
    X = X.to(device)
    y_hat = clf(X).detach().numpy()
    y = y.numpy()
    test_losses.append([
        np.mean(np.abs(y_hat - y)),
        np.sqrt(np.mean((y_hat - y)**2)),
        r2_score(y, y_hat)
    ])

test_losses = np.mean(np.nan_to_num(test_losses, neginf=0), axis=0)
print(test_losses)



import pickle as pkl
save_path = "transformer_losses.pkl"

mamba_losses = {
   'train': train_losses, 
   'val': val_losses, 
   'test': test_losses, 
}

with open(save_path, "wb") as f:
    pkl.dump(mamba_losses, f)



clf.train()
opt = torch.optim.Adam(clf.parameters(), lr=2e-4, weight_decay=1e-4)

ONLINE_EPOCHS = 4




online_val_losses = []
for X_v, y_v in tqdm(val_data, "Evaluating validation loss"):

    X_v = X_v.to(device)
    y_hat = clf(X_v).detach().numpy()
    y_v2 = y_v.numpy()
    online_val_losses.append([
        np.mean(np.abs(y_hat - y_v2)),
        np.sqrt(np.mean((y_hat - y_v2)**2)),
        r2_score(y_v2, y_hat)
    ])

    for _ in range(ONLINE_EPOCHS):
        y_hat = clf(X_v)
        loss = mae_loss(y_hat, y_v)
        opt.zero_grad()
        loss.backward()
        opt.step()

online_val_losses = np.mean(np.nan_to_num(online_val_losses, neginf=0), axis=0)
print(online_val_losses)



online_test_losses = []
for X_v, y_v in tqdm(test_data, "Evaluating test loss"):
   
    X_v = X_v.to(device)
    y_hat = clf(X_v).detach().numpy()
    y_v2 = y_v.numpy()
    online_test_losses.append([
        np.mean(np.abs(y_hat - y_v2)),
        np.sqrt(np.mean((y_hat - y_v2)**2)),
        r2_score(y_v2, y_hat)
    ])

    for _ in range(ONLINE_EPOCHS):
        y_hat = clf(X_v)
        loss = mae_loss(y_hat, y_v)
        opt.zero_grad()
        loss.backward()
        opt.step()

online_test_losses = np.mean(np.nan_to_num(online_test_losses, neginf=0), axis=0)
print(online_test_losses)