import ROOT
import uproot
import numpy as np
import sys

# Open the ROOT file
nano_file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")

# Retrieve important info data
events = nano_file["Events"]

PV_x = events["PV_x"].array(library="np", entry_stop=100)
PV_y = events["PV_y"].array(library="np", entry_stop=100)
PV_z = events["PV_z"].array(library="np", entry_stop=100)



# Function to check if an event contains exactly two muons
def contains_two_muons(event):
    muon_count = np.sum(np.abs(event) == 13)
    return muon_count == 2

# Function to apply a pT cut of 2.0 GeV
def pt_cuts(event):
    muon_indices = np.where(np.abs(event) == 13)[0]
    muon_pts = event[muon_indices]
    if len(muon_pts) == 2:  # If there are two muons
        pt1, pt2 = muon_pts
        if pt1 > 2.0 and pt2 > 2.0: 
            return True
    return False



# Filter events by containing exactly two muons
two_muon_events = [event for event in events["PFCands_pdgId"].array(library="np", entry_stop=100) if contains_two_muons(event)]

# Filter events by pT cut
two_muon_pt_cut_events = [event for event in two_muon_events if np.any(pt_cuts(event))]

# Filter events by vertex (making sure all PFs per event come from same vertex)




# Makes everything the same shape
filtered_events = [event for event in two_muon_pt_cut_events if len(event) == len(two_muon_pt_cut_events[0])]
two_muon_events_array = np.array(filtered_events)


print("Total number of events in file:", events.num_entries)
print("Total number of events with exactly 2 muons:", len(two_muon_events)) 
print("Total number of events with exactly 2 muons and pT > 2 GeV:", len(two_muon_pt_cut_events))
print("Type of two_muon_events_array:", type(two_muon_events_array))
print("Shape of two_muon_events_array:", two_muon_events_array.shape)
