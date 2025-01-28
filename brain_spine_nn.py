import tensorflow as tf
import numpy as np

# Loading the split data
def load_split(load_dir="./data_preprocessing/split_data/"):
    X_train = np.load(f"{load_dir}X_train.npy")
    X_test = np.load(f"{load_dir}X_test.npy")
    y_binary_train = np.load(f"{load_dir}y_binary_train.npy")
    y_binary_test = np.load(f"{load_dir}y_binary_test.npy")
    y_obesity_train = np.load(f"{load_dir}y_obesity_train.npy")
    y_obesity_test = np.load(f"{load_dir}y_obesity_test.npy")
    
    return X_train, X_test, y_binary_train, y_binary_test, y_obesity_train, y_obesity_test

X_train, X_test, y_binary_train, y_binary_test, y_obesity_train, y_obesity_test=load_split()

# Define the small network
def build_small_network(input_shape):
    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # gives binary output regarding healthy weight
    return tf.keras.Model(inputs, outputs, name="SmallNetwork")


def build_large_network(input_shape, small_output_shape):
    # Inputs
    inputs = tf.keras.Input(shape=(input_shape,))
    small_output = tf.keras.Input(shape=(small_output_shape,))
    
    # Main path
    x = tf.keras.layers.Dense(64, activation="relu")(inputs) 
    x = tf.keras.layers.Dense(32, activation="leaky_relu")(x)
    x = tf.keras.layers.Concatenate()([x, small_output])    # spine network output passed in middle of hidden layers     
    x = tf.keras.layers.Dense(16, activation="tanh")(x) 
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)  # Multi-class classification for obesity

    return tf.keras.Model([inputs, small_output], outputs, name="LargeNetwork")

# Build and compile both networks
small_network = build_small_network(X_train.shape[1])
large_network = build_large_network(X_train.shape[1], 1)

small_network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
large_network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Train the small network
print("Training the small network...")
small_network.fit(X_train, y_binary_train, validation_split=0.2, epochs=10, batch_size=32)

# Get small network outputs for the large network
small_train_outputs = small_network.predict(X_train)
small_test_outputs = small_network.predict(X_test)


# Train the large network
print("Training the large network...")
large_network.fit([X_train, small_train_outputs], y_obesity_train, validation_split=0.2, epochs=25, batch_size=32)

# Evaluate the networks
print("\nEvaluating the small network...")
small_loss, small_acc = small_network.evaluate(X_test, y_binary_test)
print(f"Small Network Accuracy: {small_acc:.2f}")

print("\nEvaluating the large network...")
large_loss, large_acc = large_network.evaluate([X_test, small_test_outputs], y_obesity_test)
print(f"Large Network Accuracy: {large_acc:.2f}")

small_network.save("small_network.h5")
large_network.save("large_network.h5")

