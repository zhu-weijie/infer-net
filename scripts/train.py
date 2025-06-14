import tensorflow as tf
import os

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "infer-net.keras")
IMAGE_SIZE = (28, 28)
IMAGE_SHAPE = IMAGE_SIZE + (1,)
BATCH_SIZE = 32
EPOCHS = 5

print("Loading Fashion MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((-1,) + IMAGE_SHAPE)
x_test = x_test.reshape((-1,) + IMAGE_SHAPE)
print(f"Data loaded and preprocessed. Training shape: {x_train.shape}")


print("Building the model...")
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=IMAGE_SHAPE),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()


print("Training the model...")
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
)


print("Evaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")


print("Saving the trained model...")
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
