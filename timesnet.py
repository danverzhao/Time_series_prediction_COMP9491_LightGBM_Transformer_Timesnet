import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from sklearn.metrics import r2_score

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.num_kernels = configs.num_kernels

        self.conv = nn.Sequential(
            Inception_Block_V1(self.d_model, self.d_ff, num_kernels=self.num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.d_ff, self.d_model, num_kernels=self.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.shape
        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # residual connection
        res = res + x
        return res

    def FFT_for_Period(self, x, k):
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesNet(nn.Module):
    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TimesNet
        for i in range(len(self.model)):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # project back
        dec_out = self.projection(enc_out)
        
        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

# Configuration
class Config:
    task_name = 'short_term_forecast'
    seq_len = 5
    label_len = 2
    pred_len = 1
    top_k = 3
    num_kernels = 4
    d_model = 500
    d_ff = 500
    e_layers = 3
    enc_in = 17
    c_out = 1
    embed = 'fixed'
    freq = 's'
    dropout = 0.05
    

# Model initialization
config = Config()
model = TimesNet(config)

# Function to prepare decoder input
def create_decoder_input(y):
    batch_size = y.shape[0]
    dec_inp = torch.zeros(batch_size, config.pred_len + config.label_len, 17)
    dec_inp[:, :config.label_len, :] = y[:, :config.label_len, :]
    return dec_inp


def r_squared(y_true, y_pred):
    # Reshape arrays to (100,) to simplify calculations
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Calculate total sum of squares
    # ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # # Calculate residual sum of squares
    # ss_res = np.sum((y_true - y_pred)**2)
    
    # # Calculate R-squared
    # r2 = 1 - (ss_res / ss_tot)
    
    return r2_score(y_true, y_pred)

# Example usage (training loop)
def train(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_x, batch_y, batch_x_mark in train_loader:
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x_mark = batch_x_mark.to(device)
        
        # Prepare additional inputs
        batch_y_mark = batch_x_mark[:, -config.pred_len:, :]
        dec_inp = create_decoder_input(batch_y).to(device)
        
        # Forward pass
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs[:, :, 0].unsqueeze(-1)  # Only use the first feature
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
       

    # validation
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        val_loss = 0
        for batch_x, batch_y, batch_x_mark in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            
            batch_y_mark = batch_x_mark[:, -config.pred_len:, :]
            dec_inp = create_decoder_input(batch_y).to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, :, 0].unsqueeze(-1)  # Only use the first feature
            val_loss += criterion(outputs, batch_y).item()

            all_outputs.append(outputs.cpu())
            all_targets.append(batch_y.cpu())

        val_loss /= len(val_loader)

        # Calculate additional metrics
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        mae = np.mean(np.abs(all_outputs - all_targets))
        rmse = np.sqrt(np.mean((all_outputs - all_targets)**2))
        r2 = r_squared(all_targets, all_outputs)
    
    print(f"Total training time: {(time.time() - start_time):.2f}s")
    return total_loss / len(train_loader), val_loss, mae, rmse, r2

# Example main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet(Config()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss() 

    # Assuming you have your data loaders set up
    # train_loader = ...
    #=======================================================================================
    dir = 'grouped_data/5_days'

    X_train = np.load(f'{dir}/X_train.npy')
    y_train = np.load(f'{dir}/y_train.npy')
    y_train = y_train.flatten()

    print('train sets:')
    print(X_train.shape)
    print(y_train.shape)

    X_test = np.load(f'{dir}/X_test.npy') 
    y_test = np.load(f'{dir}/y_test.npy') 
    y_test = y_test.flatten()

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1).unsqueeze(-1)  # Shape: [samples, 1, 1]
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(-1).unsqueeze(-1)  # Shape: [samples, 1, 1]

    # Create time feature tensors
    train_mark = torch.arange(config.seq_len).repeat(X_train.shape[0], 1)
    train_mark = train_mark.unsqueeze(-1).repeat(1, 1, 4).float()
    test_mark = torch.arange(config.seq_len).repeat(X_test.shape[0], 1)
    test_mark = test_mark.unsqueeze(-1).repeat(1, 1, 4).float()

    # Function to prepare decoder input
    def create_decoder_input(y):
        batch_size = y.shape[0]
        dec_inp = torch.zeros(batch_size, config.pred_len + config.label_len, 17)
        dec_inp[:, :config.label_len, :] = y[:, :config.label_len, :]
        return dec_inp

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_mark)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_mark)

    # Create DataLoaders
    batch_size = 1024  # You can adjust this value

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print information about the loaders
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")

    # Example of iterating through the train_loader
    for batch_x, batch_y, batch_x_mark in train_loader:
        batch_y_mark = batch_x_mark[:, -config.pred_len:, :]
        dec_inp = create_decoder_input(batch_y)
        
        print(f"Train batch X shape: {batch_x.shape}")
        print(f"Train batch y shape: {batch_y.shape}")
        print(f"Train batch x_mark shape: {batch_x_mark.shape}")
        print(f"Train batch y_mark shape: {batch_y_mark.shape}")
        print(f"Train batch dec_inp shape: {dec_inp.shape}")
        break  # Just print the first batch and break
    #=======================================================================================

    num_epochs = 500
    for epoch in range(num_epochs):
        loss, val_loss, mae, rmse, r2 = train(model, train_loader, val_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()