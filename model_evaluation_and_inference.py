from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load the trained model
model = load_model("D:\\projects and shit\\ChessBot\\trained_chess_model.h5")


print("=== Model Loaded ===")

# Define data directories and parameters
base_dir = "D:\\projects and shit\\ChessBot"
batch_size = 16
img_height = 128
img_width = 128

# Create ImageDataGenerator for data validation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 20% of data will be used for validation
)

# Generate validation data
val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

print("=== Starting Model Evaluation ===")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

print("=== Model Evaluation Complete ===")

# Inference

print("=== Starting Inference ===")

# Load a new image file for prediction
img_path = "D:\\projects and shit\\ChessBot\\Pawn\\Pawn_Skin1_Black_001.jpg"
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Get the class label (e.g., "Pawn", "Rook", etc.)
class_label = list(val_gen.class_indices.keys())[predicted_class]
print(f"Predicted Class: {class_label}")

print("=== Inference Complete ===")
