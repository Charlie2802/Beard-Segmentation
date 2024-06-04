import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image
from keras.metrics import MeanIoU

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 16

# Directories
train_images_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\train_2\train\images'
train_masks_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\train_2\train\masks'
val_images_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\\train_2\val\images'
val_masks_dir = r'C:\Users\jarvis\Downloads\FINAL_BEARD_DATASET_UPDATED\train_2\val\masks'

# Function to check if an image is valid
def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, Image.DecompressionBombError, Image.UnidentifiedImageError):
        return False

# Data generator class
class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, img_size):
        self.image_paths = [path for path in image_paths if is_image_valid(path)]
        self.mask_paths = [path for path in mask_paths if is_image_valid(path)]
        self.batch_size = batch_size
        self.img_size = img_size
        self.on_epoch_end()

    def __len__(self):
        # Ensure at least one batch is returned even if there are fewer images than the batch size
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_image_paths = self.image_paths[start_index:end_index]
        batch_mask_paths = self.mask_paths[start_index:end_index]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            image = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0

            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]

# Function to create the U-Net model

# U-Net model architecture
import tensorflow as tf

# U-Net model architecture with UpSampling2D
import tensorflow as tf

# U-Net model architecture with UpSampling2D
def unet_model(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.1)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.1)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.1)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.UpSampling2D((2, 2))(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.UpSampling2D((2, 2))(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.UpSampling2D((2, 2))(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.UpSampling2D((2, 2))(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    # Create and return the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Load image and mask paths
def load_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if is_image_valid(os.path.join(image_dir, fname))])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if is_image_valid(os.path.join(mask_dir, fname))])
    return image_paths, mask_paths

# Load training and validation data paths
train_image_paths, train_mask_paths = load_image_mask_paths(train_images_dir, train_masks_dir)
val_image_paths, val_mask_paths = load_image_mask_paths(val_images_dir, val_masks_dir)

# Ensure non-empty validation data
assert len(val_image_paths) > 0, "Validation image paths are empty!"
assert len(val_mask_paths) > 0, "Validation mask paths are empty!"

# Create data generators
train_generator = DataGenerator(train_image_paths, train_mask_paths, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))
val_generator = DataGenerator(val_image_paths, val_mask_paths, BATCH_SIZE, (IMG_WIDTH, IMG_HEIGHT))



def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + 1.0) / (sum_ - intersection + 1.0)
    return 1 - tf.reduce_mean(jac)

# Create U-Net model
model = unet_model()

# Compile the model with binary cross-entropy (BCE) and Jaccard loss, and metrics including accuracy and MeanIoU
model.compile(optimizer='adam', loss=['binary_crossentropy', jaccard_loss], metrics=[ MeanIoU(num_classes=2)])

# Train the model
steps_per_epoch = len(train_image_paths) // BATCH_SIZE
validation_steps = len(val_image_paths) // BATCH_SIZE

num_epochs = 20 # Specify the number of epochs
# train_generator = train_generator.repeat(num_epochs)
# val_generator = val_generator.repeat(num_epochs)

# Now, you can use the modified generators in the model.fit() function
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=num_epochs
)


# Save the model
model.save('beard_segmentation_model_224.h5')
from keras import backend as K
K.clear_session()