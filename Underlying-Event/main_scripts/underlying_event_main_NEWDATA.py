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

file_index_path = "/app/Underlying-Event/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt"
output_direcotry = "/app/Underlying-Event/Underlying-Event/"
tree_name = "Events"
branches = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"
]

iteration_chuck_size = 1000 # Number of events per chunk
min_muon_pT = 20.0          # Minimum transverse momentum for a muon
z_mass_range = (85.0, 95.0)  # Z candidate mass window (GeV)
dz_threshold = 0.1          # Maximum allowed difference in muon vertex z positions
max_number_Non_Muons = 200      # Maximum number of non-muon particles per event
particle_features = 4   # Number of features per particle (pt, eta, phi, mass)

#########################################
# Event Processing Function
#########################################
def process_events(data):
    """
    Process a chunk of events to select Z candidate events and extract features from non-muon candidates.
    Returns:
        A tuple containing:
        - event_inputs: List of feature lists for each event.
        - event_targets: List of target values (sum of muon Pz) for each event.
        - invariant_masses: List of invariant masses of the selected Z candidates.
        - z_pt_list: List of Z candidate transverse momenta.
        - z_pz_list: List of Z candidate Pz values.
        - z_phi_list: List of Z candidate azimuthal angles.
        - z_eta_list: List of Z candidate pseudorapidities.
    """
    event_inputs, event_targets = [], []
    invariant_masses = []  
    z_pt_list, z_pz_list, z_phi_list, z_eta_list = [], [], [], []

    particle_ids = data[branches[0]]
    pts        = data[branches[1]]
    etas       = data[branches[2]]
    phis       = data[branches[3]]
    masses     = data[branches[4]]
    vertices_z = data[branches[5]]
    
    total_events = len(particle_ids)
    selected_events = 0  # Counter for events passing selection cuts

    # Loop over each event in the chunk of data
    for evt_ids, evt_pts, evt_etas, evt_phis, evt_masses, evt_vertices in zip(
        particle_ids, pts, etas, phis, masses, vertices_z):
        muon_vectors = []
        muon_charges = []
        muon_vertex_z = []

        # Loop over each candidate in the event
        for pdgid, pt, eta, phi, mass, vertex in zip(evt_ids, evt_pts, evt_etas, evt_phis, evt_masses, evt_vertices):
            if abs(pdgid) == 13 and pt > min_muon_pT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt, eta, phi, mass)
                muon_vectors.append(tlv)
                muon_charges.append(np.sign(pdgid))
                muon_vertex_z.append(vertex)

        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue
        if abs(muon_vertex_z[0] - muon_vertex_z[1]) > dz_threshold:
            continue

        # Calculate the invariant mass of the Z candidate (sum of two muons)
        z_candidate = muon_vectors[0] + muon_vectors[1]
        z_mass = z_candidate.M()
        if not (z_mass_range[0] <= z_mass <= z_mass_range[1]):
            continue

        invariant_masses.append(z_mass)
        z_pt_list.append(z_candidate.Pt())
        z_pz_list.append(z_candidate.Pz())
        z_phi_list.append(z_candidate.Phi())
        z_eta_list.append(z_candidate.Eta())

        # Compute sum of Pz from the muon pair (our target)
        muon_pz_sum = muon_vectors[0].Pz() + muon_vectors[1].Pz()

        nonmuon_features = []
        for pdgid, pt, eta, phi, mass in zip(evt_ids, evt_pts, evt_etas, evt_phis, evt_masses):
            if abs(pdgid) == 13:
                continue
            nonmuon_features.extend([pt, eta, phi, mass])

        # Limit number of non-muon particles and pad if needed
        num_particles = len(nonmuon_features) // particle_features
        event_features = []
        for i in range(min(num_particles, max_number_Non_Muons)): 
            start = i * particle_features
            event_features.extend(nonmuon_features[start:start + particle_features])
        required_length = max_number_Non_Muons * particle_features
        event_features.extend([0.0] * (required_length - len(event_features)))

        event_inputs.append(event_features)
        event_targets.append(muon_pz_sum)
        selected_events += 1

    print("Selected events in this chunk:", selected_events)
    return (event_inputs, event_targets, invariant_masses, 
            z_pt_list, z_pz_list, z_phi_list, z_eta_list)

def process_chunk(chunk_data):
    """
    Wrapper to process a chunk of data. Intended to be run in parallel  using 
    ProcessPoolExecutor. This is a python function for parallel execution of tasks.

    I think this ProcessPoolExecutor itself is a wrapper for python MPI. Need to confirm this though.... 
    """
    return process_events(chunk_data)


#########################################
# Save Processed Events to CSV
#########################################
def save_events_to_csv(inputs, targets, filename="filtered_events.csv"):
    """
    I am saving to CSV so that I only push the particles I care about to the DNN.
    I could use contunine to use the AOD format, but I am not sure how to do that currently.
    """
    num_features = max_number_Non_Muons * particle_features
    # Create column names  for features
    columns = [f"feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(inputs, columns=columns)
    df["target"] = targets
    csv_path = os.path.join(output_diretory, filename)
    df.to_csv(csv_path, index=False)
    print("Saved filtered events to:", csv_path)

#########################################
# Model Build and Evaluation tpye functions
#########################################
def build_model(input_dim):
    """
    Builds and compiles a simple regression model using TensorFlow.
    """
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        tf.keras.layers.Dense(1000),
        tf.keras.layers.Dense(100),
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
    filepath = os.path.join(output_diretory, filename)
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

#########################################
# Main Execution: Load, Process, and Train
#########################################
def main():
    num_input_files = 1  # Adjust this to select more ROOT files
    with open(file_index_path) as f:
        root_files = [line.strip() for line in f if line.strip()]

    file_map = {file: tree_name for file in root_files}
    
    chunk_iterator = uproot.iterate(
        files=file_map,
        expressions=branches,
        library="ak",
        step=iteration_chuck_size
    )

    all_event_inputs = []
    all_event_targets = []

    # Process each chunk in parallel using  ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunk_iterator]
        for future in as_completed(futures):
            try:
                result = future.result()
                event_inputs, event_targets, *_ = result
                all_event_inputs.extend(event_inputs)
                all_event_targets.extend(event_targets)
            except Exception as error:
                print("Error during chunk processing:", error)

    if not all_event_inputs:
        raise ValueError("No events passed the Z selection criteria in any chunk.")

    csv_filename = "filtered_Z_events.csv"
    save_events_to_csv(all_event_inputs, all_event_targets, filename=csv_filename)
    df = pd.read_csv(os.path.join(output_diretory, csv_filename))
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = build_model(input_dim=max_number_Non_Muons * particle_features)

    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2, validation_split=0.2) 
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)
    
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred)
    
    plot_predictions(y_test, y_pred)
    plot_training_loss(history)

if __name__ == '__main__':
    main()