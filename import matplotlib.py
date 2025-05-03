import matplotlib.pyplot as plt

# Dummy data resembling a normal training curve
epochs = list(range(1, 21))
train_accuracy = [0.50, 0.60, 0.68, 0.72, 0.76, 0.80, 0.83, 0.85, 0.87, 0.89, 0.90, 0.91, 0.915, 0.92, 0.925, 0.927, 0.930, 0.933, 0.935, 0.937]
val_accuracy = [0.48, 0.58, 0.66, 0.70, 0.73, 0.77, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.885, 0.89, 0.891, 0.892, 0.893, 0.894, 0.895, 0.896]
train_loss = [1.2, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.30, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]
val_loss = [1.3, 1.1, 0.9, 0.75, 0.65, 0.58, 0.52, 0.49, 0.45, 0.42, 0.4, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.305, 0.3]

# Plot
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("/mnt/data/training_plot.png")
plt.show()

