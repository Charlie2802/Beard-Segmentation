import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
IMG_SIZE = 224
NUM_CLASSES = 1  # Binary classification for beard/no beard

# Directories
train_dir = 'path_to_train_images'
train_mask_dir = 'path_to_train_masks'
val_dir = 'path_to_val_images'
val_mask_dir = 'path_to_val_masks'

# Load and preprocess data
def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image / 255.0

def load_and_preprocess_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask[..., np.newaxis]  # add channel dimension
    return mask / 255.0

def create_dataset(img_folder, mask_folder):
    img_paths = [os.path.join(img_folder, fname) for fname in os.listdir(img_folder)]
    mask_paths = [os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)]
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), load_and_preprocess_mask(y)))
    return dataset

# DeepLabV3+ model with Xception
def DeeplabV3Plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=input_shape))

    # Encoder
    layer_names = [
        'block4_sepconv2_bn',  # Output size: IMG_SIZE/4
        'block13_sepconv2_bn',  # Output size: IMG_SIZE/16
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]

    # ASPP
    x = layers.Conv2D(256, 3, padding="same", dilation_rate=(12, 12), activation="relu")(base_model.output)
    x = layers.BatchNormalization()(x)

    # Decoder
    x = layers.UpSampling2D(4, interpolation='bilinear')(x)
    x = Concatenate()([x, layers_output[1]])
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(4, interpolation='bilinear')(x)
    x = Concatenate()([x, layers_output[0]])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(4, interpolation='bilinear')(x)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Create and compile model
model = DeeplabV3Plus()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Load datasets
train_dataset = create_dataset(train_dir, train_mask_dir).batch(8).prefetch(2)
val_dataset = create_dataset(val_dir, val_mask_dir).batch(8).prefetch(2)

# Model training
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save model
model.save('beard_segmentation_model.h5')
