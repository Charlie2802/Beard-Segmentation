from tensorflow.keras.models import load_model
import os
import tensorflow as tf

def jaccard_loss(y_true, y_pred):
    # Example implementation, replace with your actual function
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = (intersection + 1e-15) / (sum_ - intersection + 1e-15)
    return 1 - jac

# Assuming jaccard_loss is already defined or imported
custom_objects = {"jaccard_loss": jaccard_loss}

# Ensure the path to the models directory is correct
models_path = 'MODELS'  # Set the directory containing the models
models = os.listdir(models_path)

for model_name in models:
    model_path = os.path.join(models_path, model_name)
    print("Loading model from:", model_path)

    try:
        # Load the model with the custom loss function
        model = load_model(model_path, custom_objects=custom_objects)
        model.save(model,include_optimizer=False)
        model.summary()
    except Exception as e:
        print("Failed to load model from:", model_path)
        print("Error:", str(e))
