import re
import matplotlib.pyplot as plt

# Function to extract metrics from the text
def extract_metrics(text):
    epochs = []
    mae_values = []
    rmse_values = []
    r2_values = []
    
    for line in text.split('\n'):
        if line.startswith('Epoch'):
            match = re.search(r'Epoch (\d+)', line)
            if match:
                epochs.append(int(match.group(1)))
        elif line.startswith('MAE'):
            metrics = re.findall(r'(MAE|RMSE|R\^2): ([-\d.]+)', line)
            for metric, value in metrics:
                if metric == 'MAE':
                    mae_values.append(float(value))
                elif metric == 'RMSE':
                    rmse_values.append(float(value))
                elif metric == 'R^2':
                    r2_values.append(float(value))
    
    return epochs, mae_values, rmse_values, r2_values

# The text containing the metrics
text = """
Epoch 1/500, Loss: 6.4532, Validation Loss: 6.0257
MAE: 6.0273, RMSE: 9.3509, R^2: -0.1096
Total training time: 100.48s
Epoch 2/500, Loss: 6.4519, Validation Loss: 6.0187
MAE: 6.0203, RMSE: 9.2715, R^2: -0.0908
Total training time: 128.74s
Epoch 3/500, Loss: 6.4436, Validation Loss: 6.0241
MAE: 6.0257, RMSE: 9.3971, R^2: -0.1205
Total training time: 86.47s
Epoch 4/500, Loss: 6.4411, Validation Loss: 6.0171
MAE: 6.0187, RMSE: 9.3268, R^2: -0.1038
Total training time: 85.59s
Epoch 5/500, Loss: 6.4403, Validation Loss: 6.0107
MAE: 6.0123, RMSE: 9.2884, R^2: -0.0948
Total training time: 92.11s
Epoch 6/500, Loss: 6.4393, Validation Loss: 6.0109
MAE: 6.0125, RMSE: 9.2985, R^2: -0.0971
Total training time: 89.02s
Epoch 7/500, Loss: 6.4393, Validation Loss: 6.0101
MAE: 6.0118, RMSE: 9.2910, R^2: -0.0954
Total training time: 94.72s
Epoch 8/500, Loss: 6.4384, Validation Loss: 6.0112
MAE: 6.0128, RMSE: 9.2750, R^2: -0.0916
Total training time: 88.90s
Epoch 9/500, Loss: 6.4351, Validation Loss: 6.0062
MAE: 6.0078, RMSE: 9.2384, R^2: -0.0830
Total training time: 93.04s
Epoch 10/500, Loss: 6.4325, Validation Loss: 6.0165
MAE: 6.0181, RMSE: 9.3587, R^2: -0.1114
Total training time: 88.40s
Epoch 11/500, Loss: 6.4299, Validation Loss: 6.0037
MAE: 6.0053, RMSE: 9.2548, R^2: -0.0869
Total training time: 86.82s
Epoch 12/500, Loss: 6.4290, Validation Loss: 6.0083
MAE: 6.0099, RMSE: 9.2826, R^2: -0.0934
Total training time: 85.05s
Epoch 13/500, Loss: 6.4266, Validation Loss: 6.0049
MAE: 6.0065, RMSE: 9.2647, R^2: -0.0892
Total training time: 92.55s
Epoch 14/500, Loss: 6.4252, Validation Loss: 6.0101
MAE: 6.0117, RMSE: 9.3396, R^2: -0.1069
Total training time: 89.08s
Epoch 15/500, Loss: 6.4242, Validation Loss: 6.0053
MAE: 6.0069, RMSE: 9.2367, R^2: -0.0826
Total training time: 86.59s
Epoch 16/500, Loss: 6.4226, Validation Loss: 6.0137
MAE: 6.0153, RMSE: 9.3562, R^2: -0.1108
Total training time: 89.24s
Epoch 17/500, Loss: 6.4219, Validation Loss: 6.0105
MAE: 6.0121, RMSE: 9.3398, R^2: -0.1069
Total training time: 92.44s
Epoch 18/500, Loss: 6.4208, Validation Loss: 6.0244
MAE: 6.0261, RMSE: 9.3926, R^2: -0.1195
Total training time: 89.47s
Epoch 19/500, Loss: 6.4200, Validation Loss: 6.0083
MAE: 6.0099, RMSE: 9.3042, R^2: -0.0985
Total training time: 86.35s
Epoch 20/500, Loss: 6.4192, Validation Loss: 6.0235
MAE: 6.0252, RMSE: 9.4051, R^2: -0.1224
Total training time: 84.88s
Epoch 21/500, Loss: 6.4181, Validation Loss: 5.9999
MAE: 6.0015, RMSE: 9.2706, R^2: -0.0906
Total training time: 92.34s
Epoch 22/500, Loss: 6.4171, Validation Loss: 6.0068
MAE: 6.0084, RMSE: 9.2724, R^2: -0.0910
Total training time: 89.98s
Epoch 23/500, Loss: 6.4165, Validation Loss: 5.9993
MAE: 6.0009, RMSE: 9.2660, R^2: -0.0895
Total training time: 91.68s
Epoch 24/500, Loss: 6.4193, Validation Loss: 6.0046
MAE: 6.0062, RMSE: 9.2525, R^2: -0.0863
Total training time: 84.67s
Epoch 25/500, Loss: 6.4263, Validation Loss: 6.0179
MAE: 6.0195, RMSE: 9.3397, R^2: -0.1069
Total training time: 91.97s
Epoch 26/500, Loss: 6.4234, Validation Loss: 6.0041
MAE: 6.0058, RMSE: 9.2860, R^2: -0.0942
Total training time: 90.56s
Epoch 27/500, Loss: 6.4222, Validation Loss: 6.0097
MAE: 6.0113, RMSE: 9.3277, R^2: -0.1040
Total training time: 86.97s
Epoch 28/500, Loss: 6.4214, Validation Loss: 6.0052
MAE: 6.0068, RMSE: 9.2588, R^2: -0.0878
Total training time: 84.60s
Epoch 29/500, Loss: 6.4205, Validation Loss: 6.0112
MAE: 6.0129, RMSE: 9.3379, R^2: -0.1065
Total training time: 116.89s
Epoch 30/500, Loss: 6.4196, Validation Loss: 6.0054
MAE: 6.0070, RMSE: 9.3036, R^2: -0.0984
"""

# Extract the metrics
epochs, mae_values, rmse_values, r2_values = extract_metrics(text)

# Create the plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plot MAE
ax1.plot(epochs, mae_values, 'b-o')
ax1.set_title('Mean Absolute Error (MAE)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MAE')

# Plot RMSE
ax2.plot(epochs, rmse_values, 'r-o')
ax2.set_title('Root Mean Square Error (RMSE)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('RMSE')

# Plot R^2
ax3.plot(epochs, r2_values, 'g-o')
ax3.set_title('R-squared (R^2)')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('R^2')

plt.subplots_adjust(hspace=0.4)
# plt.tight_layout()
plt.show()