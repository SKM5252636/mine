import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt

# Class names
class_names = ['Cats', 'Dogs']  # Update with actual class names if different

# Training directory
train_dir = r'C:\Users\LENOVO\PycharmProjects\nn\train'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split for validation
)

# Training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(class_names), activation='softmax')(x)

# Build model
transfer_model = models.Model(inputs=base_model.input, outputs=predictions)
transfer_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Summary
transfer_model.summary()

# Train the model
print("Training started...")
history = transfer_model.fit(train_generator, validation_data=val_generator, epochs=10)
print("Training completed.")

# Save the model
print("Saving the model...")
transfer_model.save(r'C:\Users\LENOVO\PycharmProjects\nn\transfer_learning_resnet50_model.h5')
print("Model saved successfully.")

# Make a prediction on a single image
print("Making predictions...")
img_path = r'C:\Users\LENOVO\PycharmProjects\nn\pet.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict
predictions = transfer_model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100
predicted_class_name = class_names[predicted_class]

# Show image with prediction
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {predicted_class_name} ({confidence:.2f}%)')
plt.show()
print("Prediction completed.")
