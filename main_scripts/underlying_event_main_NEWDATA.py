#!/usr/bin/env python3

import uproot
import awkward as ak
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ROOT import TLorentzVector
 
 #OpenCMS Data link: https://opendata.cern.ch/record/24110

#########################################
# Configuration and Selection Parameters
#########################################
data_file = "/nfs/home/avendras/Underlying-Event/root_files/002DAE91-77A7-E511-B61B-00266CFAEA48.root"
tree_name = "Events"
print("reading in data now:")
branches = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"

    # General tracks? Something like:
    # recoTracks_generalTracks__RECO./recoTracks_generalTracks__RECO.obj/recoTracks_generalTracks__RECO.obj.vertex_.fCoordinates.fZ
    #   
]

entry_stop = 1000


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
    Prints all available branch names in the ROOT file.
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        branches = tree.keys()
        print("Available branches in the ROOT file:")
        for branch in branches:
            print("  ", branch)


def load_data(file_path: str, tree_name: str, branches: list):#, entry_stop: int):
    """
    Loads the specified branches from the ROOT file as Awkward Arrays.
    
    Returns:
        data (dict): Dictionary containing Awkward Arrays for each branch.
    """
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        data = tree.arrays(branches,library="ak")# entry_stop=entry_stop, library="ak")
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
    pdgId_array = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_"]
    pt_array    = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt"]
    eta_array   = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta"]
    phi_array   = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi"]
    mass_array  = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM"]
    vertex_array = data["recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"]

    # Iterate over events (Awkward arrays support iteration)
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
        # "continue" skips the event if not passed.
        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue

        # Check if the muons are from the same vertex using dz threshold.
        # "continue" skips the event if not passed.
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
                continue  # Skip muons. Should probably make this less general to not skip ALL muons.
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

    # Return the processed inputs, targets, invariant masses, and Z candidate properties
    return event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list

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
    tf.keras.utils.set_random_seed(42) 
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
    list_available_branches(data_file, tree_name)
    data = load_data(data_file, tree_name, branches)#, entry_stop)

    # Process events and obtain Z candidate properties
    event_inputs, event_targets, invariant_masses, z_pt_list, z_pz_list, z_phi_list, z_eta_list = process_events(data)
    if not event_inputs:
        raise ValueError("No events passed the Z selection criteria.")

    X = np.array(event_inputs)
    y = np.array(event_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    model = build_model(input_dim=max_num_nonMuons * features_per_particle)
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2, validation_split=0.2)
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error on Test Set:", mse)
    y_pred = model.predict(X_test)
    evaluate_correlation(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    plot_training_loss(history)
    plot_loss(history)

    # Plots the invariant mass distribution
    counts, bin_edges = np.histogram(invariant_masses, bins=50, range=(50, 150))
    peak_index = np.argmax(counts)
    peak_mass = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2

    plt.figure()
    plt.hist(invariant_masses, bins=100, range=(0, 150), histtype='step', color='blue')
    plt.xlabel('Dimuon Invariant Mass [GeV]')
    plt.ylabel('Number of Events')
    plt.title('Dimuon Invariant Mass Distribution')
    plt.axvline(x=peak_mass, color='red', linestyle='--', label=f'Peak: {peak_mass: } GeV')
    plt.legend()
    plt.savefig("dimuon_mass.png")
    print("Plot saved: dimuon_mass.png")
    plt.close()

    # plot Z candidate properties

    # Plot Z pT distribution
    plt.figure()
    plt.hist(z_pt_list, bins=50, color='green', histtype='step')
    plt.xlabel("Z Candidate pT [GeV]")
    plt.ylabel("Number of Events")
    plt.title("Z Candidate pT Distribution")
    plt.savefig("z_pt_distribution.png")
    print("Plot saved: z_pt_distribution.png")
    plt.close()

    # Plot Z pz distribution
    plt.figure()
    plt.hist(z_pz_list, bins=50, color='orange', histtype='step')
    plt.xlabel("Z Candidate pz [GeV]")
    plt.ylabel("Number of Events")
    plt.title("Z Candidate pz Distribution")
    plt.savefig("z_pz_distribution.png")
    print("Plot saved: z_pz_distribution.png")
    plt.close()

    # Plot Z phi distribution
    plt.figure()
    plt.hist(z_phi_list, bins=50, color='purple', histtype='step')
    plt.xlabel("Z Candidate Phi [rad]")
    plt.ylabel("Number of Events")
    plt.title("Z Candidate Phi Distribution")
    plt.savefig("z_phi_distribution.png")
    print("Plot saved: z_phi_distribution.png")
    plt.close()

    # Plot Z eta distribution
    plt.figure()
    plt.hist(z_eta_list, bins=50, color='red', histtype='step')
    plt.xlabel("Z Candidate Eta")
    plt.ylabel("Number of Events")
    plt.title("Z Candidate Eta Distribution")
    plt.savefig("z_eta_distribution.png")
    print("Plot saved: z_eta_distribution.png")
    plt.close()

if __name__ == '__main__':
    main()
