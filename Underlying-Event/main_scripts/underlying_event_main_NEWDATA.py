#!/usr/bin/env python3

import os
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from ROOT import TLorentzVector

#########################################
# Configuration
#########################################
FILE_INDEX_PATH = "/app/Underlying-Event/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt"
#check 
OUTPUT_DIR = "/app/Underlying-Event/Underlying-Event/"

# Data and tree configuration of branches I'm interested in
TREE_NAME = "Events"
BRANCHES = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"
]
ENTRY_STOP = None

# Event selection parameters
MIN_MUON_PT = 20.0           # Minimum muon pT in GeV
Z_MASS_WINDOW = (85.0, 95.0)   # Z candidate mass window in GeV
DZ_THRESHOLD = 0.1           # Max allowed dz difference between muons
MAX_NUM_NONMUONS = 200       # Maximum non-muon candidates per event
FEATURES_PER_PARTICLE = 4    # [pt, eta, phi, mass]
MAX_FILES_TO_READ = 1     # Limit the number of files processed

#########################################
# File Loading Functions
#########################################
def load_file_paths(file_path):
    """
    Load file paths from a text file with one per line.
    """
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_data_files():
    """
    Retrieve and limit the list of data file paths.
    """
    all_paths = load_file_paths(FILE_INDEX_PATH)
    return all_paths[:MAX_FILES_TO_READ]

#########################################
# Data Loading and Debugging
#########################################
def list_available_branches(file_path, tree_name):
    """
    Print all available branches in a given ROOT file.
    """
    try:
        print("Opening file:", file_path)
        with uproot.open(file_path, timeout=18000) as file:
            tree = file[tree_name]
            print("Available branches:")
            for branch in tree.keys():
                print("  ", branch)
    except Exception as e:
        print("Error opening file:", e)

def load_data(file_paths, tree_name, branches, entry_stop=None):
    """
    Load and concatenate data from multiple ROOT files.
    """
    files_dict = {fp: tree_name for fp in file_paths}
    data = uproot.concatenate(files_dict, expressions=branches, library="ak")
    if entry_stop is not None:
        data = ak.Array({branch: data[branch][:entry_stop] for branch in branches})
    return data

#########################################
# Data Processing Function
#########################################
def process_events(data):
    """
     Prcess events to select Z candidates and extract features of non-muon canidates.
     The selection cuts for the events are as follows:
     - Select events with exactly two muons.
     - Muon pair charges must be opposite.
     - Muon pT > 20 GeV.
     - Muon pair invariant mass within 85-95 GeV 
     - Muon pair dz difference < 0.1 cm
    """
    event_inputs, event_targets = [], []
    invariant_masses = []  
    z_pt_list, z_pz_list, z_phi_list, z_eta_list = [], [], [], []

    pdgId_array = data[BRANCHES[0]]
    pt_array    = data[BRANCHES[1]]
    eta_array   = data[BRANCHES[2]]
    phi_array   = data[BRANCHES[3]]
    mass_array  = data[BRANCHES[4]]
    vertex_array = data[BRANCHES[5]]

    total_events = len(pdgId_array)
    total_selected_events = 0
    print("Total events in file:", total_events)

    for event in zip(pdgId_array, pt_array, eta_array, phi_array, mass_array, vertex_array):
        event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex = event
        muon_vectors, muon_charges, muon_vertex_z = [], [], []

        for pdgId, pt_val, eta_val, phi_val, mass_val, vertex_z in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex
        ):
            if abs(pdgId) == 13 and pt_val > MIN_MUON_PT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                muon_vectors.append(tlv)
                muon_charges.append(np.sign(pdgId))
                muon_vertex_z.append(vertex_z)

        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue
        if abs(muon_vertex_z[0] - muon_vertex_z[1]) > DZ_THRESHOLD:
            continue

        z_candidate = muon_vectors[0] + muon_vectors[1]
        z_mass = z_candidate.M()
        if not (Z_MASS_WINDOW[0] <= z_mass <= Z_MASS_WINDOW[1]):
            continue

        invariant_masses.append(z_mass)
        z_pt_list.append(z_candidate.Pt())
        z_pz_list.append(z_candidate.Pz())
        z_phi_list.append(z_candidate.Phi())
        z_eta_list.append(z_candidate.Eta())

        muon_pz_sum = muon_vectors[0].Pz() + muon_vectors[1].Pz()

        nonmuon_features = []
        for pdgId, pt_val, eta_val, phi_val, mass_val in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass
        ):
            if abs(pdgId) == 13:
                continue
            nonmuon_features.extend([pt_val, eta_val, phi_val, mass_val])

        num_nonmuons = len(nonmuon_features) // FEATURES_PER_PARTICLE
        event_vector = []
        for i in range(min(num_nonmuons, MAX_NUM_NONMUONS)): 
            start = i * FEATURES_PER_PARTICLE
            event_vector.extend(nonmuon_features[start:start + FEATURES_PER_PARTICLE]) #Keras expects fixed-length flat vectors
        event_vector.extend([0.0] * (MAX_NUM_NONMUONS * FEATURES_PER_PARTICLE - len(event_vector)))

        event_inputs.append(event_vector)
        event_targets.append(muon_pz_sum)
        total_selected_events += 1

    print("Selected events:", total_selected_events)
    return event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list

def save_events_to_csv(event_inputs, event_targets, filename="filtered_events.csv"):
    """
    Save filtered events to a CSV file.
    
    Parameters:
      event_inputs: List of lists containing event features.
      event_targets: List containing event target values.
      filename: Name of the CSV file to save.
    """
    num_features = MAX_NUM_NONMUONS * FEATURES_PER_PARTICLE
    
    columns = [f"feature_{i}" for i in range(num_features)]
    
    df = pd.DataFrame(event_inputs, columns=columns)
    
    df["target"] = event_targets
    df.to_csv(filename, index=False)
    print("Saved filtered events to:", filename)



#########################################
# Model Building and Evaluation
#########################################
def build_model(input_dim):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_correlation(y_true, y_pred):
    y_pred = np.array(y_pred).flatten()
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    print("Pearson Correlation Coefficient:", correlation)
    return correlation

#########################################
# Visualization Functions
#########################################
def save_plot(plt_obj, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt_obj.savefig(filepath)
    print("Plot saved:", filepath)
    plt_obj.close()

def plot_predictions(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Labels (Sum of Muon Pz)')
    plt.ylabel('Predicted Labels')
    plt.title('True vs. Predicted Labels')
    save_plot(plt, "prediction_main.png")

def plot_training_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    save_plot(plt, "loss_plot.png")

#########################################
# Main Execution
#########################################
def main():
    data_files = get_data_files()
    list_available_branches(data_files[0], TREE_NAME)
    
    data = load_data(data_files, TREE_NAME, BRANCHES, entry_stop=ENTRY_STOP)
    results = process_events(data)
    event_inputs, event_targets, _, _, _, _, _ = results
    if not event_inputs:
        raise ValueError("No events passed the Z selection criteria.")

    csv_filename = "filtered_Z_events.csv"
    save_events_to_csv(event_inputs, event_targets, csv_filename)

    df = pd.read_csv(csv_filename)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = build_model(input_dim=MAX_NUM_NONMUONS * FEATURES_PER_PARTICLE)
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2, validation_split=0.2)
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)
    
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    plot_training_loss(history)

if __name__ == '__main__':
    main()
