import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


MODEL_PATH = "action_violence_model.h5"
IMG_SIZE = 96
BATCH_SIZE = 32
DATA_DIR = "processed_dataset"


print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")



datagen = ImageDataGenerator(rescale=1./255)

data_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("[INFO] Running evaluation...")
preds = model.predict(data_gen, verbose=1)

y_pred = np.argmax(preds, axis=1)
y_true = data_gen.classes

class_labels = list(data_gen.class_indices.keys())


report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels
)

print("\n==============================")
print("Classification Report:")
print("==============================")
print(report)


with open("classification_report.txt", "w") as f:
    f.write(report)


f1 = f1_score(y_true, y_pred, average="weighted")
print(f"\nWeighted F1 Score: {f1:.4f}")


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Violence Detection Model")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n[INFO] Confusion matrix saved as 'confusion_matrix.png'")
print("[INFO] Classification report saved as 'classification_report.txt'")
