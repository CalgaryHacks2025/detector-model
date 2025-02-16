import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0005

# Load train data to get class names
raw_train_data = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
class_names = raw_train_data.class_names
logger.info(f"Detected class names: {class_names}")

# Optimize Dataset Loading
AUTOTUNE = tf.data.AUTOTUNE
train_data = raw_train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_data = tf.keras.utils.image_dataset_from_directory(
    'data/validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
).cache().prefetch(buffer_size=AUTOTUNE)

test_data = tf.keras.utils.image_dataset_from_directory(
    'data/test',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
).cache().prefetch(buffer_size=AUTOTUNE)

# Model Architecture
logger.info("Building the model...")
model = Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Global Average Pooling instead of Flatten
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),

    Dense(len(class_names), activation='softmax')
])

# Compile the Model
logger.info("Compiling the model with learning rate: %.6f", LEARNING_RATE)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Add Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the Model
logger.info("Starting model training...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stopping])

# Evaluate the Model
logger.info("Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(test_data)
logger.info(f"Test Accuracy: {test_acc:.2f}")

# Calculate AUROC
logger.info("Calculating AUROC metrics...")
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images)
    y_pred.append(predictions)
    y_true.append(labels.numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Compute AUROC per class
auroc_per_class = roc_auc_score(tf.keras.utils.to_categorical(y_true, len(class_names)), y_pred, average=None)
mean_auroc = roc_auc_score(tf.keras.utils.to_categorical(y_true, len(class_names)), y_pred, average='macro')

logger.info("AUROC per class:")
for class_name, auroc in zip(class_names, auroc_per_class):
    logger.info(f"AUROC for {class_name}: {auroc:.2f}")
logger.info(f"Mean AUROC: {mean_auroc:.2f}")

# Generate Classification Report
y_pred_classes = np.argmax(y_pred, axis=1)
logger.info("\n" + classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
logger.info(f"Confusion Matrix:\n{conf_matrix}")

# Save the Model
model.save('animal_image_detector_improved.h5')
logger.info("Model training complete. Model saved as animal_image_detector_improved.h5")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('improved_training_accuracy.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('improved_training_loss.png')
plt.show()
