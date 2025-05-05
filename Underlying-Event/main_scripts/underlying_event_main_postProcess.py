#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

output_dir = "/afs/cern.ch/user/a/avendras/work/Underlying-Event/Underlying-Event/plots/"
csv_file = "filtered_Z_events_80.csv"

def build_model(input_dim):
    # Basic DNN model with some regularization + dropout to prevent overfitting
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(200, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(200))
   

    model.add(tf.keras.layers.Dense(70, kernel_regularizer=regularizers.l2(1e-5)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(10, kernel_regularizer=regularizers.l2(1e-5)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(1))  # output layer (linear-regression)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mean_squared_error')
    return model


def plot_predictions(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, s=2, alpha=0.7)
    plt.xlabel("Actual (Sum of Muon Pz)")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_main_new.png")
    plt.savefig(path)
    print(f"Saved scatter plot to {path}")
    plt.close()


def plot_loss_curve(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_plot_new.png")
    plt.savefig(path)
    print(f"Saved loss curve to {path}")
    plt.close()


def evaluate_corr(y_true, y_pred):
    # y_true and y_pred should be in original units
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"Pearson Corr: {corr:.4f}")
    return corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                        default=os.path.join(output_dir, csv_file),
                        help="Path to your CSV file with data")

    args = parser.parse_args()
    csv_path = args.csv_path

    print(f"Reading from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events")

    # Split features and target
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # Scale target to [0,1]. THis seems to help with overfitting issues
    y_max = y.max()
    y_scaled = y / y_max
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

    # Evaluate on test set (scaled MSE)
    test_loss_scaled = model.evaluate(X_test, y_test, verbose=0)
 
    test_loss = test_loss_scaled * (y_max ** 2)   # converts MSE back to original units: MSE scales by (y_max)^2
    print(f"Test MSE (original units): {test_loss:.3f}")

    y_pred_scaled = model.predict(X_test).flatten() # Predict and rescale predictions
    y_pred = y_pred_scaled * y_max
    y_true = y_test * y_max

    # Evaluate and plot in original units
    evaluate_corr(y_true, y_pred)
    plot_predictions(y_true, y_pred)
    plot_loss_curve(history)

if __name__ == "__main__":
    main()
