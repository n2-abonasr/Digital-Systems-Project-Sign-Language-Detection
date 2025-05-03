import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint

# Load my landmarks dataset
DATASET_PATH = 'dataset/landmarks.csv'  
data = pd.read_csv(DATASET_PATH)

# Separate features (X) and labels (y)
X = data.iloc[:, 1:].values   # all the landmark coordinates
y = data.iloc[:, 0].values    # the letter labels (A, B, C, etc)

# Encode labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build a simple model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best model automatically
checkpoint = ModelCheckpoint(
    'best_landmarks_model.keras', 
    monitor='val_accuracy', 
    mode='max', 
    save_best_only=True,
    verbose=1
)

# Train.
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,         # More epochs = better accuracy
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

print(" Training complete! Best model saved as best_landmarks_model.keras")
