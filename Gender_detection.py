import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB7
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint  # Import ModelCheckpoint

# Define data directories
train_dir = 'men_women_classification/train_data'  # Path to training data
validation_dir = 'men_women_classification/validation'  # Path to validation data
test_dir = 'men_women_classification/test'  # Path to test data

# Image size for EfficientNetB7
img_height, img_width = 600, 600

# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split training data into 80% training and 20% validation
)

# Load and preprocess data using data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,  # Adjust batch size as needed
    class_mode='categorical',  # Two classes, so categorical
    subset='training'  # Specify that this is for training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=32,  # Adjust batch size as needed
    class_mode='categorical',  # Two classes, so categorical
    subset='validation'  # Specify that this is for validation data
)

# Define a ModelCheckpoint callback
checkpoint_path = "/Reasearch_Fashion_recommend/model_checkpoint.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',  # You can choose the metric to monitor
    save_best_only=True,  # Save only the best models
    mode='max',  # For accuracy, use 'max'; for loss, use 'min'
    verbose=1
)

# Load pre-trained EfficientNetB7 model
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)  # Two classes: 'Men' and 'Women'

# Create the custom model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the checkpoint callback
epochs = 10  # Adjust the number of epochs as needed
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[checkpoint]  # Include the checkpoint callback here
)

# Save the model for later use
model.save('gender_detection_efficientnetb7.h5')
