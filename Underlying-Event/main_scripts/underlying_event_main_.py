#!/usr/bin/env python3
"""
Cleaned and Organized Script for ROOT Data Processing and Neural Network Training
"""

import uproot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ROOT import TLorentzVector

#########################################
# Configuration and Selection Parameters
#########################################
data_file = "/nfs/home/avendras/Underlying-Event/root_files/002DAE91-77A7-E511-B61B-00266CFAEA48.root"
tree_name = "Events"
branches = [
    "PFCands_pdgId",
    "PFCands_pt",
    "PFCands_eta",
    "PFCands_phi",
    "PFCands_mass",
    "PFCands_dz"
]
entry_stop = 1


# Event selection parameters
min_Muon_pT = 20.0              # Minimum muon pT (GeV)
z_mass_window = (85.0, 95.0)    # Allowed invariant mass window for Z candidate (GeV)
dZ_threshold = 0.1              # Maximum allowed dz difference for same vertex
max_num_nonMuons = 100          # Maximum number of non-muon candidates per event
features_per_particle = 4       # Features per candidate: [pt, eta, phi, mass]


#########################################
# Utility/Debugging Functions
#########################################
def list_available_branches(file_path: str, tree_name: str) -> None:
    """
    Prints all available branch names in the ROOT file.
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        branches = tree.keys()
        print("Available branches in the ROOT file:")
        for branch in branches:
            print("  ", branch)


def load_data(file_path: str, tree_name: str, branches: list, entry_stop: int):
    """
    Loads the specified branches from the ROOT file as numpy arrays.
    
    Returns:
        data (dict): Dictionary containing numpy arrays for each branch.
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        data = tree.arrays(branches, entry_stop=entry_stop, library="np")
    return data


#########################################
# Data Processing Function
#########################################
def process_events(data):
    """
    Processes events by applying Z candidate selection criteria and 
    constructing input vectors (non-muon features) and target values (sum of muon pz).
    
    Returns:
        event_inputs (list): Flattened input vectors for non-muon candidates.
        event_targets (list): Target values corresponding to the sum of muon pz.
    """
    event_inputs = []
    event_targets = []

    pdgId_array = data["PFCands_pdgId"]
    pt_array = data["PFCands_pt"]
    eta_array = data["PFCands_eta"]
    phi_array = data["PFCands_phi"]
    mass_array = data["PFCands_mass"]
    dz_array = data["PFCands_dz"]

    for event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_dz in zip(
        pdgId_array, pt_array, eta_array, phi_array, mass_array, dz_array
    ):
        muon_vectors = []
        muon_dzs = []
        muon_charges = []

        # Collect muon candidates passing the pT cut
        for pdgId, pt_val, eta_val, phi_val, mass_val, dz_val in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_dz
        ):
            if abs(pdgId) == 13 and pt_val > min_Muon_pT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                muon_vectors.append(tlv)
                muon_dzs.append(dz_val)
                muon_charges.append(np.sign(pdgId))

        # opposite charges citeria
        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue

        # invariant mass of the muon pair (Z candidate)
        z_candidate = muon_vectors[0] + muon_vectors[1]
        z_mass = z_candidate.M()
        if not (z_mass_window[0] <= z_mass <= z_mass_window[1]):
            continue

        # making sure hte muons come from the same vertex
        if abs(muon_dzs[0] - muon_dzs[1]) > dZ_threshold:
            continue

        # Use the average dz of the muons as the vertex
        z_dz = np.mean(muon_dzs)
        muon_pz_sum = muon_vectors[0].Pz() + muon_vectors[1].Pz()

        # I try to make a dz threshold to make the particles come from the same vertex
        nonmuon_features = []
        for pdgId, pt_val, eta_val, phi_val, mass_val, dz_val in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_dz
        ):
            if abs(pdgId) == 13:
                continue  # Skip muons
            if abs(dz_val - z_dz) > dZ_threshold:
                continue
            nonmuon_features.extend([pt_val, eta_val, phi_val, mass_val])

        # Build a fixed-length input vector: select up to max_num_nonMuons candidates and pad if needed
        num_nonmuons = len(nonmuon_features) // features_per_particle
        event_vector = []
        for i in range(min(num_nonmuons, max_num_nonMuons)):
            start = i * features_per_particle
            event_vector.extend(nonmuon_features[start:start + features_per_particle])
        # Pad with zeros if there are fewer than max_num_nonMuons candidates
        while len(event_vector) < max_num_nonMuons * features_per_particle:
            event_vector.extend([0.0] * features_per_particle)

        event_inputs.append(event_vector)
        event_targets.append(muon_pz_sum)

    return event_inputs, event_targets


#########################################
# Neural Network Model Definition
#########################################
def build_model(input_dim: int):
    """
    Builds and compiles the Keras Sequential model.
    
    Args:
        input_dim: Dimension of the input vector.
        
    Returns:
        model: Compiled Keras model.
    """
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(1000, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Dense(1000, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dense(550, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#########################################
# Data Evaluation Functions
#########################################
def evaluate_correlation(y_true, y_pred):
    """
    Evaluates the Pearson correlation coefficient between the true and predicted values.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    
    Returns:
        float: Pearson correlation coefficient.
    """
    # Ensure y_pred is a 1D array in case it's in shape (n,1)
    y_pred = np.array(y_pred).flatten()
    
    # Calculate the correlation matrix and extract the correlation coefficient
    corr_matrix = np.corrcoef(y_true, y_pred)
    correlation = corr_matrix[0, 1]
    
    print("Pearson Correlation Coefficient:", correlation)
    return correlation



#########################################
# Visualization Functions
#########################################
def plot_predictions(y_true, y_pred, filename="prediction_main.png"):
    """
    Plots a scatter plot comparing true vs. predicted values.
    """
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Labels (Sum of Muon Pz)')
    plt.ylabel('Predicted Labels')
    plt.title('True vs. Predicted Labels')
    plt.savefig(filename)
    print("plot saved: prediction_main.png")
    plt.close()


def plot_training_loss(history, filename="model_loss_main.png"):
    """
    Plots the training loss over epochs.
    """
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(filename)
    print("plot saved: model_loss_main.png")
    plt.close()

def plot_loss(history, filename="loss_plot.png"):
    """
    Plots training loss and validation loss over epochs on the same plot.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')

    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved: {filename}")
    plt.close()


#########################################
# Main Execution Function
#########################################
def main():
    # List available branches for reference and loads the local root file 
    list_available_branches(data_file, tree_name)
    data = load_data(data_file, tree_name, branches, entry_stop)

    # Process events and apply selection criteria
    event_inputs, event_targets = process_events(data)
    if not event_inputs:
        raise ValueError("No events passed the Z selection criteria.")

    # Convert lists to numpy arrays and split data into training and testing sets
    X = np.array(event_inputs)
    y = np.array(event_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Build and train the neural network model
    model = build_model(input_dim=max_num_nonMuons * features_per_particle)
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2, validation_split=0.2)

    # Evaluate the model on the test set
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)

    # Visualize predictions and training loss
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    plot_training_loss(history)
    plot_loss(history)


if __name__ == '__main__':
    main()