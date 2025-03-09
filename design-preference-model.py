import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, classification_report

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5  # minimalist, colorful, retro, sleek, futuristic

# Set paths to dataset (to be updated with actual paths)
data_dir = "design_preferences_dataset/"

# Data preprocessing and augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Create datasets
train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Create a test dataset with a separate ImageDataGenerator (no augmentation)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# For testing, assuming you have a separate test folder
test_ds = test_datagen.flow_from_directory(
    data_dir + 'test/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Create the base model from the pre-trained MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = MobileNetV2(input_shape=IMG_SHAPE,
                        include_top=False,
                        weights='imagenet')

# Freeze the base model first
base_model.trainable = False

# Create the model
def create_model():
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

model = create_model()

# Compile the model
base_learning_rate = 0.0001
model.compile(
    optimizer=optimizers.Adam(learning_rate=base_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model (Phase 1: train only the top layers)
initial_epochs = 10
history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds
)

# Plot learning curves
def plot_learning_curves(history, initial_epochs=None):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if initial_epochs:
        plt.plot([initial_epochs-1, initial_epochs-1], 
                plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if initial_epochs:
        plt.plot([initial_epochs-1, initial_epochs-1], 
                plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.title('Training and Validation Loss')
    plt.savefig('learning_curves_phase1.png')
    plt.show()

# Plot the learning curves for Phase 1
plot_learning_curves(history)

# Phase 2: Fine-tuning
# Unfreeze the top layers of the base model
fine_tune_at = 140  # Number of layers to freeze in the base model
base_model.trainable = True

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=base_learning_rate/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training the model
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=val_ds
)

# Plot the learning curves for both phases
history_combined = history.history.copy()
for key in history_fine.history:
    history_combined[key] = history.history[key] + history_fine.history[key]

plot_learning_curves(history_combined, initial_epochs)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_ds)
print(f'Test accuracy: {test_accuracy:.4f}')

# Generate predictions for the test set
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_ds.classes

# Generate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(cm)

# Generate classification report
class_names = list(test_ds.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print('Classification Report:')
print(report)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save the model
model.save('design_preferences_model.h5')
print('Model saved successfully.')

# Function to make predictions on new inputs
def predict_design_preference(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    class_names = list(train_ds.class_indices.keys())
    result = {
        'class': class_names[predicted_class],
        'confidence': float(confidence)
    }
    
    return result
