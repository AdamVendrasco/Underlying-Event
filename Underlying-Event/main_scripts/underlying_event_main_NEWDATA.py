#!/usr/bin/env python3

import uproot
import awkward as ak
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ROOT import TLorentzVector

# OpenCMS Data link: https://opendata.cern.ch/record/24110

#########################################
# Configuration and Selection Parameters
#########################################
print("Reading in data now:")
print("LOOK FOR ME")
data_files = [
    "/app/Underlying-Event/root_files/002DAE91-77A7-E511-B61B-00266CFAEA48.root"
    "/app/Underlying-Event/root_files/002DAE91-77A7-E511-B61B-00266CFAEA48.root",
    "/app/Underlying-Event/root_files/008D888F-80A7-E511-B17F-0CC47A78A4A0.root",
    "/app/Underlying-Event/root_files/006E50B5-6BA7-E511-AB89-7845C4FC374C.root",
    "/app/Underlying-Event/root_files/002ADEBA-30A7-E511-A6B2-0CC47A4C8E66.root"
]
output_dir = "/app/Underlying-Event/plots/"

tree_name = "Events"
print("Reading in data from multiple files now:")
branches = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"
]
entry_stop = None  

# Event selection parameters
min_Muon_pT = 20.0              # Minimum muon pT (GeV)
z_mass_window = (85.0, 95.0)    # Allowed invariant mass window for Z candidate (GeV)
dZ_threshold = 0.1              # Maximum allowed dz difference for same vertex
max_num_nonMuons = 200          # Maximum number of non-muon candidates per event
features_per_particle = 4       # Features per candidate: [pt, eta, phi, mass]

#########################################
# Utility/Debugging Functions
#########################################
def list_available_branches(file_path: str, tree_name: str) -> None:
    """
    Prints all available branch names in a given ROOT file.
    """
    with uproot.open(file_path) as file:
        print(f"Trying to open file: {file_path}")
        tree = file[tree_name]
        branch_names = tree.keys()
        print("Available branches in the ROOT file:")
        for branch in branch_names:
            print("  ", branch)

def load_data(file_paths: list, tree_name: str, branches: list, entry_stop: int = None):
   
    files_dict = {fp: tree_name for fp in file_paths}
    data = uproot.concatenate(
        files_dict,
        expressions=branches,
        library="ak"
    )

    if entry_stop is not None:
        data = ak.Array({branch: data[branch][:entry_stop] for branch in branches})
    return data



#########################################
# Data Processing Function
#########################################
def process_events(data):
    event_inputs = []
    event_targets = []
    invariant_masses = []  

    # Lists for Z candidate properties
    z_pt_list = []
    z_pz_list = []
    z_phi_list = []
    z_eta_list = []

    # Retrieve Awkward arrays for each branch
    pdgId_array = data[branches[0]]
    pt_array    = data[branches[1]]
    eta_array   = data[branches[2]]
    phi_array   = data[branches[3]]
    mass_array  = data[branches[4]]
    vertex_array = data[branches[5]]

    total_events = len(pdgId_array)
    total_selected_events =  0
    print("Total events in file:", total_events)

    for event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex in zip(
        pdgId_array, pt_array, eta_array, phi_array, mass_array, vertex_array
    ):
        muon_vectors = []
        muon_charges = []
        muon_vertex_z = [] 

        # Collect muon candidates passing the pT cut
        for pdgId, pt_val, eta_val, phi_val, mass_val, vertex_z in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass, event_vertex
        ):
            if abs(pdgId) == 13 and pt_val > min_Muon_pT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                muon_vectors.append(tlv)
                muon_charges.append(np.sign(pdgId))
                muon_vertex_z.append(vertex_z)

        # Require exactly 2 muons with opposite charge
        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue

        # Check if the muons are from the same vertex using dz threshold.
        dz = abs(muon_vertex_z[0] - muon_vertex_z[1])
        if dz > dZ_threshold:
            continue

        # Calculate the invariant mass of the muon pair (Z candidate)
        z_candidate = muon_vectors[0] + muon_vectors[1]
        z_mass = z_candidate.M()

        # Check that the mass is within the allowed window
        if not (z_mass_window[0] <= z_mass <= z_mass_window[1]):
            continue

        invariant_masses.append(z_mass)

        # Store the Z candidate properties
        z_pt_list.append(z_candidate.Pt())
        z_pz_list.append(z_candidate.Pz())
        z_phi_list.append(z_candidate.Phi())
        z_eta_list.append(z_candidate.Eta())

        muon_pz_sum = muon_vectors[0].Pz() + muon_vectors[1].Pz()

        # Process non-muon candidates for the event
        nonmuon_features = []
        for pdgId, pt_val, eta_val, phi_val, mass_val in zip(
            event_pdgIds, event_pt, event_eta, event_phi, event_mass
        ):
            if abs(pdgId) == 13:
                continue  # Skip muons
            nonmuon_features.extend([pt_val, eta_val, phi_val, mass_val])

        # Build a fixed-length input vector: select up to max_num_nonMuons candidates and pad if needed
        num_nonmuons = len(nonmuon_features) // features_per_particle
        event_vector = []
        for i in range(min(num_nonmuons, max_num_nonMuons)):
            start = i * features_per_particle
            event_vector.extend(nonmuon_features[start:start + features_per_particle])
        while len(event_vector) < max_num_nonMuons * features_per_particle:
            event_vector.extend([0.0] * features_per_particle)

        event_inputs.append(event_vector)
        event_targets.append(muon_pz_sum)
        total_selected_events += 1
    print(f"Number of events that survive the event selection: {total_selected_events}")
    return event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list

#########################################
# Neural Network Model Definition
#########################################
def build_model(input_dim: int):
    tf.keras.utils.set_random_seed(42) 
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_dim,)),
        #tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#########################################
# Evaluation Function
#########################################
def evaluate_correlation(y_true, y_pred, filename="correlation.txt"):
    y_pred = np.array(y_pred).flatten()
    corr_matrix = np.corrcoef(y_true, y_pred)
    correlation = corr_matrix[0, 1]
    print("Pearson Correlation Coefficient:", correlation)

    # Save correlation value to a file
    correlation_file = os.path.join(output_dir, filename)
    with open(correlation_file, "w") as f:
        f.write(f"Pearson Correlation Coefficient: {correlation}\n")

    print("Correlation saved:", correlation_file)
    return correlation

#########################################
# Visualization Functions
#########################################
def plot_predictions(y_true, y_pred, filename="prediction_main.png"):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Labels (Sum of Muon Pz)')
    plt.ylabel('Predicted Labels')
    plt.title('True vs. Predicted Labels')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print("Plot saved:", filepath)
    plt.close()

def plot_training_loss(history, filename="model_loss_main.png"):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print("Plot saved:", filepath)
    plt.close()

def plot_loss(history, filename="loss_plot.png"):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print("Plot saved:", filepath)
    plt.close()

#########################################
# Main Execution Function
#########################################
def main():
    # Optionally, list branches from one file (e.g., the first file)
    list_available_branches(data_files[0], tree_name)
    
    # Load data from all files (concatenated)
    data = load_data(data_files, tree_name, branches, entry_stop=entry_stop)

    # Process events and obtain Z candidate properties
    event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list = process_events(data)
    if not event_inputs:
        raise ValueError("No events passed the Z selection criteria.")

    X = np.array(event_inputs)
    y = np.array(event_targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = build_model(input_dim=max_num_nonMuons * features_per_particle)
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2, validation_split=0.2)
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)
    
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred, filename="correlation.txt")
    plot_predictions(y_test, y_pred, filename="prediction_main.png")
    plot_training_loss(history, filename="model_loss_main.png")
    plot_loss(history, filename="loss_plot.png")

if __name__ == '__main__':
    main()
