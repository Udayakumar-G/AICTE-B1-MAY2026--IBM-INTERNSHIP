import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ===========================================
# Dataset Paths (EDIT ONLY HERE if needed)
# ===========================================
DATASET_BASE = r"C:/Users/udaya/OneDrive/Documents/dtset"

PATHS = {
    "dress": os.path.join(DATASET_BASE, "dresscode"),
    "grooming": os.path.join(DATASET_BASE, "bear"),
    "posture": os.path.join(DATASET_BASE, "posture"),
    "id": os.path.join(DATASET_BASE, "card")
}

# Quick path check
for name, path in PATHS.items():
    if not os.path.exists(path):
        print(f"[WARNING] Dataset for {name} not found at {path}")

# ===========================================
# Function to build and train classifier
# ===========================================
def build_and_train_classifier(dataset_path, model_name, img_size=(128,128), batch_size=32, epochs=10):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_gen.class_indices)

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(*img_size,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    print(f"\n[INFO] Training {model_name} model...")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Save model and class indices
    model.save(f"{model_name}_model.h5")
    with open(f"{model_name}_classes.txt", "w") as f:
        for class_name in train_gen.class_indices.keys():
            f.write(class_name + "\n")

    print(f"[INFO] {model_name} model saved!")
    return model, train_gen.class_indices

# ===========================================
# Train all four classifiers
# ===========================================
if __name__ == "__main__":
    build_and_train_classifier(PATHS["dress"], "dress", epochs=10)
    build_and_train_classifier(PATHS["grooming"], "grooming", epochs=10)
    build_and_train_classifier(PATHS["posture"], "posture", epochs=10)
    build_and_train_classifier(PATHS["id"], "id", epochs=10)
