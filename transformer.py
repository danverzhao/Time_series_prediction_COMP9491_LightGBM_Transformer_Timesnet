import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

# transformer trained with 1 stock's prices history and indicator history, price prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#========================================================================
# loading data
dir = 'grouped_data'

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

X_train_scaled = X_train
Y_train_scaled = Y_train
X_test_scaled = X_test
Y_test_scaled = Y_test

print(f'X_train.shape: {X_train_scaled.shape}')
print(f'Y_train.shape: {Y_train_scaled.shape}')
print(f'X_test.shape: {X_test_scaled.shape}')
print(f'Y_test.shape: {Y_test_scaled.shape}')

#========================================================================

class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        output = self.dropout(self.relu(self.fc(output)))
        output = self.fc2(output)
        return output
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
batch_size = 128
learning_rate = 0.0001
num_epochs = 100
d_model = 128
nhead = 8
num_layers = 4
dim_feedforward = 256
validation_split = 0.2

# Prepare data
X_train = torch.FloatTensor(X_train_scaled).to(device)
Y_train = torch.FloatTensor(Y_train_scaled).to(device)
X_test = torch.FloatTensor(X_test_scaled).to(device)
Y_test = torch.FloatTensor(Y_test_scaled).to(device)

dataset = TensorDataset(X_train, Y_train)
train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

input_dim = X_train.shape[1]
model = StockPriceTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


# Early stopping parameters
patience = 20
best_val_loss = float('inf')
counter = 0
best_model = None

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        # loss = criterion(outputs, batch_y)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            # val_loss = criterion(outputs, batch_y)
            val_loss = criterion(outputs.squeeze(), batch_y)
            total_val_loss += val_loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    scheduler.step(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_model = model.state_dict()
        torch.save(best_model, 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Inference without DataLoader
model.eval()
with torch.no_grad():
    # predictions = model(X_test).cpu().numpy()
    predictions = model(X_test).squeeze().cpu().numpy()
    true_values = Y_test.cpu().numpy()

# Calculate RMSE
mse = np.mean((predictions - true_values)**2)
rmse = np.sqrt(mse)

print(f'Test RMSE: {rmse:.4f}')
