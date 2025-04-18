#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split

# Configuration parameters (should match those used in pre processing)

# output_directory = "/app/Underlying-Event/"
output_directory = "/afs/cern.ch/user/a/avendras/work/Underlying-Event/Underlying-Event/"
csv_filename = "filtered_Z_events_30.csv"
max_number_Non_Muons = 200
particle_features = 4   # Number of features per particle

#########################################
# Model Build and Evaluation functions
#########################################
def build_model(input_dim):
    """
    Builds and compiles a simple regression model using TensorFlow.
    """
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)), 
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def evaluate_correlation(y_true, y_pred):
    """
    Calculates and prints the Pearson correlation coefficient between true and predicted values.
    """
    y_pred = np.array(y_pred).flatten()
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    print("Pearson Correlation Coefficient:", correlation)
    return correlation

def save_plot(plot_obj, filename):
    """
    Saves the current plot to a file.
    """
    filepath = os.path.join(output_directory, filename)
    plot_obj.savefig(filepath)
    print("Plot saved to:", filepath)
    plot_obj.close()

def plot_predictions(true_labels, predicted_labels):
    """
    Creates and saves a scatter plot comparing true labels and predicted labels.
    """
    plt.figure()
    plt.scatter(true_labels, predicted_labels)
    plt.xlabel('True Labels (Sum of Muon Pz)')
    plt.ylabel('Predicted Labels')
    plt.title('True vs. Predicted Labels')
    save_plot(plt, "prediction_main.png")

def plot_training_loss(history):
    """
    Plots training and validation loss over epochs.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    save_plot(plt, "loss_plot.png")



def main():
    """
     - Loads CSV (given some path parser to the CSV) with filtered events that will be given to DNN.
     - Tracks and displays the total number of events given to the DNN.
     - Trains/evaluates the model.

    """
    parser = argparse.ArgumentParser(description="Train DNN on particle data.")
    parser.add_argument('--csv_path', type=str, required=False,
                        default=os.path.join(output_directory, csv_filename),
                        help='Enter full path to the input CSV file here!')

    args = parser.parse_args()
    csv_path = args.csv_path

    df = pd.read_csv(csv_path)
    total_events = df.shape[0]
    print("CSV loaded from:", csv_path)
    print("Total events loaded:", total_events)
    
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    input_dim = max_number_Non_Muons * particle_features
    model = build_model(input_dim=input_dim)

    history = model.fit(X_train, y_train, epochs=100, batch_size=10,
                        verbose=2, validation_split=0.2)

    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    plot_training_loss(history)

if __name__ == '__main__':
    main()
