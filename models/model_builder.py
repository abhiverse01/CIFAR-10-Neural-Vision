"""
model_builder.py
CNN architecture for CIFAR-10 classification.
Features: Residual blocks, data augmentation, cosine LR decay, label smoothing.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "✈ airplane", "🚗 automobile", "🐦 bird", "🐱 cat", "🦌 deer",
    "🐶 dog", "🐸 frog", "🐴 horse", "🚢 ship", "🚚 truck"
]
CLASS_NAMES_CLEAN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
BINARY_CLASSES = {0: "✈ Airplane (Class A)", "other": "🚗 Non-Airplane (Class B)"}
IMG_SIZE = (32, 32)
NUM_CLASSES = 10


# ─── Data Augmentation Pipeline ───────────────────────────────────────────────
def get_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


# ─── Residual Block ───────────────────────────────────────────────────────────
def residual_block(x, filters, stride=1, l2=1e-4):
    shortcut = x

    x = layers.Conv2D(filters, 3, stride, padding="same",
                      kernel_regularizer=regularizers.l2(l2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, 1, padding="same",
                      kernel_regularizer=regularizers.l2(l2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Project shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, stride, padding="same",
                                 kernel_regularizer=regularizers.l2(l2), use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


# ─── Multi-Class Model (ResNet-style) ─────────────────────────────────────────
def create_multiclass_model(use_augmentation=True):
    inputs = layers.Input(shape=(32, 32, 3), name="input")
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    # Stem
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual stages
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="multiclass_out")(x)

    model = models.Model(inputs, outputs, name="ResNet_MultiClass")
    return model


# ─── Binary Model (airplane vs rest) ──────────────────────────────────────────
def create_binary_model(use_augmentation=True):
    inputs = layers.Input(shape=(32, 32, 3), name="input")
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    # Stem
    x = layers.Conv2D(64, 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual stages
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_out")(x)

    model = models.Model(inputs, outputs, name="ResNet_Binary")
    return model


# ─── Learning Rate Schedule ───────────────────────────────────────────────────
def cosine_annealing_schedule(epoch, lr, total_epochs=30, min_lr=1e-6, max_lr=1e-3):
    """Cosine annealing with warm restarts."""
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))


# ─── Training Function ────────────────────────────────────────────────────────
def train_models(epochs=30, batch_size=128, save_dir="models/"):
    import os

    print("📦 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda e, lr: cosine_annealing_schedule(e, lr, total_epochs=epochs)
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    # ── Multi-class ──
    print("\n🚀 Training Multi-Class Model...")
    mc_model = create_multiclass_model()
    mc_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    mc_history = mc_model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks, verbose=1
    )
    mc_model.save(os.path.join(save_dir, "model_multiclass.keras"))
    print(f"✅ Multi-class model saved. Best val accuracy: {max(mc_history.history['val_accuracy']):.4f}")

    # ── Binary ──
    print("\n🚀 Training Binary Model (Airplane vs Rest)...")
    y_train_binary = (y_train == 0).astype("float32")
    y_test_binary = (y_test == 0).astype("float32")

    bin_model = create_binary_model()
    bin_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    bin_history = bin_model.fit(
        x_train, y_train_binary,
        epochs=epochs, batch_size=batch_size,
        validation_data=(x_test, y_test_binary),
        callbacks=callbacks, verbose=1
    )
    bin_model.save(os.path.join(save_dir, "model_binary.keras"))
    print(f"✅ Binary model saved. Best val AUC: {max(bin_history.history['val_auc']):.4f}")

    return mc_history, bin_history


if __name__ == "__main__":
    train_models()
