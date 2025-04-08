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
from concurrent.futures import ProcessPoolExecutor, as_completed

#########################################
# Configuration
#########################################
FILE_INDEX_PATH = "/app/Underlying-Event/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt"
OUTPUT_DIR = "/app/Underlying-Event/Underlying-Event/"

TREE_NAME = "Events"
BRANCHES = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"
]

# Use step to define the chunk size when iterating (number of events per chunk)
CHUNK_SIZE = 1000

MIN_MUON_PT = 20.0
Z_MASS_WINDOW = (85.0, 95.0)
DZ_THRESHOLD = 0.1
MAX_NUM_NONMUONS = 200
FEATURES_PER_PARTICLE = 4

#########################################
# Event Processing Function
#########################################
def process_events(data):
    """
    Process events to select Z candidates and extract features from non-muon candidates.
    Returns a tuple:
       (event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list)
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
    # For debugging large runs, you can print the number of events in this chunk:
    # print(f"Processing {total_events} events in chunk...")
    
    for event in zip(pdgId_array, pt_array, eta_array, phi_array, mass_array, vertex_array):
        event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex = event
        muon_vectors, muon_charges, muon_vertex_z = [], [], []

        # Process each candidate in an event looking for muons
        for pdgId, pt_val, eta_val, phi_val, mass_val, vertex_z in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex
        ):
            if abs(pdgId) == 13 and pt_val > MIN_MUON_PT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                muon_vectors.append(tlv)
                muon_charges.append(np.sign(pdgId))
                muon_vertex_z.append(vertex_z)

        # Check for exactly two muons with opposite charge and small dz difference
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
            event_vector.extend(nonmuon_features[start:start + FEATURES_PER_PARTICLE])
        event_vector.extend([0.0] * (MAX_NUM_NONMUONS * FEATURES_PER_PARTICLE - len(event_vector)))

        event_inputs.append(event_vector)
        event_targets.append(muon_pz_sum)
        total_selected_events += 1

    # Print how many events passed the selection in this chunk:
    print("Selected events in this chunk:", total_selected_events)
    return event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list

#########################################
# Parallel Processing: Processing Chunks
#########################################
def process_chunk(chunk_data):
    """
    A wrapper to process a data chunk. This function is designed to be executed in a separate process.
    """
    return process_events(chunk_data)

#########################################
# Save Events to CSV
#########################################
def save_events_to_csv(event_inputs, event_targets, filename="filtered_events.csv"):
    num_features = MAX_NUM_NONMUONS * FEATURES_PER_PARTICLE
    columns = [f"feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(event_inputs, columns=columns)
    df["target"] = event_targets
    csv_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(csv_path, index=False)
    print("Saved filtered events to:", csv_path)

#########################################
# Model Building and Evaluation Functions
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
# Main Execution with uproot Iterator and Parallel Chunk Processing
#########################################
def main():
    # Get the ROOT file paths from the file index
    with open(FILE_INDEX_PATH, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    file_path = files[0]
    event_inputs_all = []
    event_targets_all = []
    
    # Using uproot.iterate to load the ROOT file in chunks.
    #Setting step=CHUNK_SIZE determines how many events per chunk.
    chunk_iterator = uproot.iterate(
        file_path,
        TREE_NAME,
        expressions=BRANCHES,
        library="ak",
        step=CHUNK_SIZE
    )
    
    # Use ProcessPoolExecutor to process each chunk in parallel.
    with ProcessPoolExecutor() as executor:
        futures = []
        for chunk in chunk_iterator:
  
            futures.append(executor.submit(process_chunk, chunk))
        
        # Retrieve results as they complete.
        # and unpacks results: event_inputs, event_targets
        for future in as_completed(futures):
            try:
                result = future.result()
                event_inputs, event_targets, _, _, _, _, _ = result
                event_inputs_all.extend(event_inputs)
                event_targets_all.extend(event_targets)
            except Exception as exc:
                print("Chunk processing generated an exception:", exc)
    
    if not event_inputs_all:
        raise ValueError("No events passed the Z selection criteria in any chunk.")

    # Save all events that pass event event selction to CSV before training model. 
    #maybe get rid of this we will see...  
    csv_filename = "filtered_Z_events.csv"
    save_events_to_csv(event_inputs_all, event_targets_all, filename=csv_filename)
    df = pd.read_csv(os.path.join(OUTPUT_DIR, csv_filename))
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
