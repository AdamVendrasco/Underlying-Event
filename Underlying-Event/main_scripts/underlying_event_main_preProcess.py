#!/usr/bin/env python3
import os
import uproot
import awkward as ak
import numpy as np
import pandas as pd
from ROOT import TLorentzVector

# Configuration parameters
#file_index_path = "/app/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt"
#output_directory = "/app/Underlying-Event/"
file_index_path = "/afs/cern.ch/user/a/avendras/work/Underlying-Event/Underlying-Event/CMS_Run2015D_DoubleMuon_AOD_16Dec2015-v1_10000_file_index.txt"
output_directory = "/afs/cern.ch/user/a/avendras/work/Underlying-Event/Underlying-Event/"
tree_name = "Events"
branches = [
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.pdgId_",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPt",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fEta",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fPhi",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.p4Polar_.fCoordinates.fM",
    "recoPFCandidates_particleFlow__RECO./recoPFCandidates_particleFlow__RECO.obj/recoPFCandidates_particleFlow__RECO.obj.m_state.vertex_.fCoordinates.fZ"
]

iteration_chunk_size = 10000  # Number of events per chunk
min_muon_pT = 20.0           # Minimum transverse momentum for a muon
z_mass_range = (85.0, 95.0)    # Z candidate mass window (GeV)
dz_threshold = 0.1           # Maximum allowed difference in muon vertex z positions
max_number_Non_Muons = 200   # Maximum number of non-muon particles per event
particle_features = 4        # Number of features per particle (pt, eta, phi, mass)

# Limit the number of input files (set to a positive integer)
num_input_files = 1  # For example, process only the first 5 files from the index


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

        # Process the event: extract muon candidates etc.
        muon_vectors = []
        muon_charges = []
        muon_vertex_z = []

        for pdgid, pt, eta, phi, mass, vertex in zip(evt_ids, evt_pts, evt_etas, evt_phis, evt_masses, evt_vertices):
            if abs(pdgid) == 13 and pt > min_muon_pT:
                tlv = TLorentzVector()
                tlv.SetPtEtaPhiM(pt, eta, phi, mass)
                muon_vectors.append(tlv)
                muon_charges.append(np.sign(pdgid))
                muon_vertex_z.append(vertex)

        # Apply selection criteria
        if len(muon_vectors) != 2 or (muon_charges[0] * muon_charges[1] >= 0):
            continue
        if abs(muon_vertex_z[0] - muon_vertex_z[1]) > dz_threshold:
            continue

        # Calculate the invariant mass of the Z candidate (sum of two muons)
        z_candidate = muon_vectors[0] + muon_vectors[1]
        z_mass = z_candidate.M()
        if not (z_mass_range[0] <= z_mass <= z_mass_range[1]):
            continue

        # If passed all selections, record event details
        invariant_masses.append(z_mass)
        z_pt_list.append(z_candidate.Pt())
        z_pz_list.append(z_candidate.Pz())
        z_phi_list.append(z_candidate.Phi())
        z_eta_list.append(z_candidate.Eta())

        muon_pz_sum = muon_vectors[0].Pz() + muon_vectors[1].Pz()

        nonmuon_features = []
        for pdgid, pt, eta, phi, mass in zip(evt_ids, evt_pts, evt_etas, evt_phis, evt_masses):
            if abs(pdgid) == 13:
                continue
            nonmuon_features.extend([pt, eta, phi, mass])

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

    # Print out summary info for the chunk:
    print(f"Processed {total_events} events in this chunk.")
    print(f"Selected events in this chunk: {selected_events}")
    return (event_inputs, event_targets, invariant_masses,
            z_pt_list, z_pz_list, z_phi_list, z_eta_list)


def process_chunk(chunk_data):
    """
    Simple wrapper function for sequential processing of a data chunk.
    """
    return process_events(chunk_data)


def save_events_to_csv(inputs, targets, filename="filtered_Z_events.csv"):
    """
    Save the filtered events to a CSV file.
    """
    num_features = max_number_Non_Muons * particle_features
    # Create column names for features
    columns = [f"feature_{i}" for i in range(num_features)]
    df = pd.DataFrame(inputs, columns=columns)
    df["target"] = targets
    csv_path = os.path.join(output_directory, filename)
    df.to_csv(csv_path, index=False)
    print("Saved filtered events to:", csv_path)


def main():
    # Load list of ROOT files from the file index
    with open(file_index_path) as f:
        root_files = [line.strip() for line in f if line.strip()]

    # Limit the number of files if requested
    if num_input_files > 0:
        root_files = root_files[:num_input_files]

    file_map = {file: tree_name for file in root_files}

    chunk_iterator = uproot.iterate(
        files=file_map,
        expressions=branches,
        library="ak",
        step=iteration_chunk_size
    )

    all_event_inputs = []
    all_event_targets = []

    # Process each chunk sequentially
    for chunk in chunk_iterator:
        try:
            event_inputs, event_targets, *_ = process_chunk(chunk)
            all_event_inputs.extend(event_inputs)
            all_event_targets.extend(event_targets)
        except Exception as error:
            print("Error during chunk processing:", error)

    if not all_event_inputs:
        raise ValueError("No events passed the Z selection criteria in any chunk.")

    save_events_to_csv(all_event_inputs, all_event_targets)

if __name__ == '__main__':
    main()