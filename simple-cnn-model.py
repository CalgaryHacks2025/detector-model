import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import logging
import random
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Model Configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.0005

# Function to sample dataset
def create_sampled_dataset(src_folder, dest_folder, samples_per_class):
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)

    for class_name in os.listdir(src_folder):
        class_path = os.path.join(src_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        if len(files) >= samples_per_class:
            sampled_files = random.sample(files, samples_per_class)
        else:
            sampled_files = files

        dest_class_path = os.path.join(dest_folder, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        for file in sampled_files:
            src_file = os.path.join(class_path, file)
            dest_file = os.path.join(dest_class_path, file)
            shutil.copy(src_file, dest_file)

# Adjust datasets
create_sampled_dataset('data/train', 'data/sampled_train', 1000)
create_sampled_dataset('data/validation', 'data/sampled_validation', 200)
create_sampled_dataset('data/test', 'data/sampled_test', 120)

# Load datasets with AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

# Create raw dataset to get class names
raw_train_data = tf.keras.utils.image_dataset_from_directory(
    'data/sampled_train',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
class_names = raw_train_data.class_names
logger.info(f"Detected class names: {class_names}")

# Apply cache and prefetch after extracting class names
train_data = raw_train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_data = tf.keras.utils.image_dataset_from_directory(
    'data/sampled_validation',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
).cache().prefetch(buffer_size=AUTOTUNE)

test_data = tf.keras.utils.image_dataset_from_directory(
    'data/sampled_test',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
).cache().prefetch(buffer_size=AUTOTUNE)

# Model Architecture (3 Conv Blocks)
logger.info("Building the model...")
model = Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Global Average Pooling
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.5),

    Dense(len(class_names), activation='softmax')
])

# Compile the Model
logger.info("Compiling the model with learning rate: %.6f", LEARNING_RATE)
learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=LEARNING_RATE, decay_steps=1500)
model.compile(optimizer=Adam(learning_rate=learning_rate_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

logger.info("Starting model training...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stopping, reduce_lr])

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
model.save('model.h5')
logger.info("Model training complete. Model saved as model.h5")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()
