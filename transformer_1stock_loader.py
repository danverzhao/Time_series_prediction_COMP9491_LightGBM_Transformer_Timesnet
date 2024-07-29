import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import math
import torch.nn.utils as utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 

# transformer trained with 1 stock's prices history and indicator history, price prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#========================================================================
# loading data
dir = 'grouped_data/5_days'

X_train = np.load(f'{dir}/X_train.npy')
Y_train = np.load(f'{dir}/y_train.npy')
Y_train = Y_train.flatten()

print('train sets:')
print(X_train.shape)
print(Y_train.shape)

X_test = np.load(f'{dir}/X_test.npy') 
Y_test = np.load(f'{dir}/y_test.npy') 
Y_test = Y_test.flatten()

print('test sets:')
print(X_test.shape)
print(Y_test.shape)

X_val = np.load(f'{dir}/X_val.npy') 
Y_val = np.load(f'{dir}/y_val.npy') 
Y_val = Y_val.flatten()

print('val sets:')
print(X_val.shape)
print(Y_val.shape)

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



X_train_scaled = X_train.reshape(len(X_train), -1)
Y_train_scaled = Y_train
X_test_scaled = X_test.reshape(len(X_test), -1)
Y_test_scaled = Y_test
X_val_scaled = X_val.reshape(len(X_val), -1)
Y_val_scaled = Y_val


print(f'X_train.shape: {X_train_scaled.shape}')
print(f'Y_train.shape: {Y_train_scaled.shape}')
print(f'X_test.shape: {X_test_scaled.shape}')
print(f'Y_test.shape: {Y_test_scaled.shape}')
print(f'X_val.shape: {X_val_scaled.shape}')
print(f'Y_val.shape: {Y_val_scaled.shape}')



#========================================================================

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
    

# Hyperparameters
batch_size = 1024  # Increased batch size due to large dataset
learning_rate = 0.0001
weight_decay = 1e-5   # Added weight decay
num_epochs = 100
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 128
validation_split = 0.2

# Prepare data
X_train = torch.FloatTensor(X_train_scaled).to(device)
Y_train = torch.FloatTensor(Y_train_scaled).to(device)
X_test = torch.FloatTensor(X_test_scaled).to(device)
Y_test = torch.FloatTensor(Y_test_scaled).to(device)
X_val = torch.FloatTensor(X_val_scaled).to(device)
Y_val = torch.FloatTensor(Y_val_scaled).to(device)


dataset = TensorDataset(X_train, Y_train)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

input_dim = X_train.shape[1]  
model = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)


# Load the best model
model.load_state_dict(torch.load('transformer_models/model2/best_model.pth', map_location=device))

# Inference without DataLoader
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    predictions = model(X_test).cpu().numpy()
    true_values = Y_test.numpy()

# Calculate RMSE
mse = np.mean((predictions - true_values)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - true_values))
r2 = r2_score(true_values, predictions)

print(f'Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2}')

#=======
# training loss
model.eval()
with torch.no_grad():
    X_train = X_train.to(device)
    predictions = model(X_train).cpu().numpy()
    true_values = Y_train.numpy()

# Calculate RMSE
mse = np.mean((predictions - true_values)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - true_values))
r2 = r2_score(true_values, predictions)

print(f'Train RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2}')

#=======
# validation loss
model.eval()
with torch.no_grad():
    X_val = X_val.to(device)
    predictions = model(X_val).cpu().numpy()
    true_values = Y_val.numpy()

# Calculate RMSE
mse = np.mean((predictions - true_values)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - true_values))
r2 = r2_score(true_values, predictions)

print(f'Val RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2}')