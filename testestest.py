import uproot
import numpy as np

def process_file(file_path):
    # Open the ROOT file
    nano_file = uproot.open(file_path)

    # Retrieve important info data
    events = nano_file["Events"]
    events.show()
    PV_x = events["PV_x"].array(library="np", entry_stop=10)
    PV_y = events["PV_y"].array(library="np", entry_stop=10)
    PV_z = events["PV_z"].array(library="np", entry_stop=10)

    # Function to check if an event contains exactly two muons
    def contains_two_muons(event):
        muon_count = np.sum(np.abs(event) == 13)
        return muon_count == 2

    # Function to apply a pT cut of 2.0 GeV
    def pt_cuts(event):
        muon_indices = np.where(np.abs(event) == 13)[0]
        muon_pts = event[muon_indices]
        pt1, pt2 = muon_pts
        if pt1 > 2.0 and pt2 > 2.0: 
            return True
        return False

    # Filter events based on pdgid (making sure 2 muons exactly and taking those events)
    two_muon_events = [event for event in events["PFCands_pdgId"].array(library="np", stop_entry=10) if contains_two_muons(event)]

    # Filter events by pT cut
    two_muon_pt_cut_events = [event for event in two_muon_events if np.any(pt_cuts(event))]

    # Iterate over events to separate muons and other particles
    muons_and_others = []
    for event in two_muon_pt_cut_events:
        muon_indices = np.where(np.abs(event) == 13)[0]
        muon_pts = event[muon_indices]
        muons = []
        others = []
        for i, pdgid in enumerate(event):
            if i in muon_indices:
                muons.append((i, pdgid, muon_pts[np.where(muon_indices == i)[0][0]]))
            else:
                if pdgid != 13:  # Check if the particle is not a muon
                    others.append((i, pdgid))
        # Ensure both muons and others lists have the same length for each event
        min_length = min(len(muons), len(others))
        muons = muons[:min_length]
        others = others[:min_length]
        muons_and_others.append((muons, others))

    # Convert muons_and_others to a Numpy array
    max_length = max(len(muons_and_others[i][0]) + len(muons_and_others[i][1]) for i in range(len(muons_and_others)))
    muons_and_others_array = np.zeros((len(muons_and_others), max_length, 3), dtype=np.int32)
    
    for i, event_tuple in enumerate(muons_and_others):
        muons, others = event_tuple
        muons_and_others_array[i, :len(muons), :] = np.array(muons)
        muons_and_others_array[i, len(muons):len(muons)+len(others), :] = np.array(others)

    print("Total number of events in file:", events.num_entries)
    print("Total number of events with exactly 2 muons:", len(two_muon_events)) 
    print("Total number of events with exactly 2 muons and pT > 2 GeV:", len(two_muon_pt_cut_events))
    print("Shape of muons_and_others_array:", muons_and_others_array.shape)

    return PV_x, PV_y, PV_z, muons_and_others_array

# Call the function with the file path
PV_x, PV_y, PV_z, muons_and_others_array = process_file("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")

