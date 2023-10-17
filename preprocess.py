from keras.preprocessing.image import ImageDataGenerator

image_size = (224, 224)  # Adjust the size as needed
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\navya\Documents\Everything\Skills_development\Go_projects\Fashion_real_code\Fashion_recommender\train_data',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # for gender classification
)

validation_generator = test_datagen.flow_from_directory(
    r'C:\Users\navya\Documents\Everything\Skills_development\Go_projects\Fashion_real_code\Fashion_recommender\validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # for gender classification
)
