# %%
pip install matplotlib

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# %%
DATA_DIR = "./Mushrooms"
IMAGE_SIZE = (128, 128) 
BATCH_SIZE = 32
NUM_CLASSES = 9
EPOCHS = 30

# %%
def load_dataset(subset):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset=subset
    ).map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE).apply(tf.data.experimental.ignore_errors())

# %%
train_ds = load_dataset('training')
val_ds = load_dataset('validation')
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# %%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.1),
])


# %%
for image, _ in train_ds.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[13]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

# %%
def create_custom_cnn():
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs)

# %%
model = create_custom_cnn()
model.summary()

# %%
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# %%
model.save("classifier.keras")
print("Model saved as classifier.keras")

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# %%
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# %%
def create_feature_extractor():
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inputs, x)  

# %%
def extract_features_labels(model, dataset):
    features = []
    labels = []
    for batch_images, batch_labels in dataset:
        feats = model.predict(batch_images)
        features.append(feats)
        labels.append(batch_labels)
    return np.vstack(features), np.vstack(labels)

# %%
pip install scikit-learn

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import joblib  

# Create feature extractor model
feature_extractor = create_feature_extractor()
feature_extractor.summary()

# %%
# Extract features
X_train, y_train = extract_features_labels(feature_extractor, train_ds)
X_val, y_val = extract_features_labels(feature_extractor, val_ds)

# Convert one-hot labels to class indices
y_train_classes = np.argmax(y_train, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

# Define the parameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 5000, 10000]
}

# Initialize the base LinearSVC model
base_svm = LinearSVC()

# Apply GridSearchCV to find the best hyperparameters
grid = GridSearchCV(base_svm, param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train_classes)

# Best model
svm = grid.best_estimator_

# Save the best model
joblib.dump(svm, 'svm_classifier.pkl')
print("SVM classifier saved as svm_classifier.pkl")
print("Best parameters:", grid.best_params_)


# %%
feature_extractor.save("feature_extractor.keras")

# %%
# Predict
y_pred = svm.predict(X_val)

# Evaluate accuracy
acc = accuracy_score(y_val_classes, y_pred)
print(f"SVM Classifier Accuracy: {acc:.4f}")


