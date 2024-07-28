
text = '''
Epoch [1/500], Train Loss: 90.4610, Val Loss: 6.4154
Epoch [2/500], Train Loss: 90.4501, Val Loss: 6.4154
Epoch [3/500], Train Loss: 90.4485, Val Loss: 6.4154
Epoch [4/500], Train Loss: 90.4485, Val Loss: 6.4154
Epoch [5/500], Train Loss: 90.4485, Val Loss: 6.4153
Epoch [6/500], Train Loss: 90.4484, Val Loss: 6.4154
Epoch [7/500], Train Loss: 90.4484, Val Loss: 6.4154
'''

import re
import matplotlib.pyplot as plt

def parse_losses(text):
    lines = text.strip().split('\n')
    train_losses = []
    val_losses = []
    epochs = []

    for line in lines:
        match = re.search(r'Epoch \[(\d+)/\d+\], Train Loss: ([\d.]+), Val Loss: ([\d.]+)', line)
        if match:
            epoch, train_loss, val_loss = match.groups()
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    return epochs, train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses_separately(epochs, train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Training Loss Plot
    ax1.plot(epochs, train_losses, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.grid(True)

    # Validation Loss Plot
    ax2.plot(epochs, val_losses, 'r-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss over Epochs')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


epochs, train_losses, val_losses = parse_losses(text)
# plot_losses(epochs, train_losses, val_losses)
plot_losses_separately(epochs, train_losses, val_losses)