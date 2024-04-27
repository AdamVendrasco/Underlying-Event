import numpy as np
import ROOT
import uproot
from ROOT import TLorentzVector

######################################################
# Open the ROOT file and retrieves important info data
######################################################

nano_file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
events = nano_file["Events"]
pf_pt = events["PFCands_pt"].array(entry_stop=50)
pf_phi = events["PFCands_phi"].array(entry_stop=50)
pf_eta = events["PFCands_eta"].array(entry_stop=50)
pf_mass = events["PFCands_mass"].array(entry_stop=50)
pf_dz = events["PFCands_dz"].array(entry_stop=50)

######################################################
# Definitions for the cuts on the data
######################################################

# Function to check if an event contains exactly two muons
def contains_two_muons(event):
    muon_count = np.sum(np.abs(event) == 13)
    return muon_count == 2

# Function to apply a pT cut of 2.0 GeV
def pt_cuts(event):
    muon_indices = np.where(np.abs(event) == 13)
    muon_pts = event[muon_indices]
    pt1, pt2 = muon_pts
    if pt1 > 2.0 and pt2 > 2.0:
        return True
    return False

# Function to filter PF muons
def filter_pf_muons(event):
    return [pfcand for pfcand in event if abs(pfcand) == 13]

# Function to filter other PF candidates and pad arrays to a fixed length
def filter_other_particles(event, max_length):
    other_particles = [pfcand for pfcand in event if abs(pfcand) != 13]
    padded_particles = other_particles + [0] * (max_length - len(other_particles))
    return padded_particles

######################################################
# Start of the cuts on the data
######################################################

# Filter events based on pdgid (making sure 2 muons exactly and taking those events)
two_muon_events = [event for event in events["PFCands_pdgId"].array(entry_stop=50) if contains_two_muons(event)]

# Filter events by pT cut
two_muon_pt_cut_events = [event for event in two_muon_events if np.any(pt_cuts(event))]



# Filter events to separate PF muons and other particles of those 2 muon events
# Also making sure the event list has the same length so I can "pad" the arrays later
filtered_muon_events = []
for event in two_muon_pt_cut_events:
    filtered_event = filter_pf_muons(event)
    filtered_muon_events.append(filtered_event)


# Calculate the maximum length of other particles in filtered_muon_events
max_other_length = max(len(event) for event in filtered_muon_events)



# Filter other particles in each event in two_muon_pt_cut_events and store in filtered_other_events
filtered_other_events = []
for event in two_muon_pt_cut_events:
    filtered_event = filter_other_particles(event, max_other_length)
    filtered_other_events.append(filtered_event)


# Recalculate the maximum length of other particles in filtered_other_events
max_other_length = max(len(event) for event in filtered_other_events)


# Pad all lists inside filtered_other_events to have the same length
filtered_other_events_padded = []
for event in filtered_other_events:
    padded_event = filter_other_particles(event, max_other_length)
    filtered_other_events_padded.append(padded_event)

# Convert each padded event into a numpy array individually
other_events_array = []
other_events_pt_array=[]
for i in filtered_other_events_padded:
    numpy_event = np.array(event)
    other_events_array.append(numpy_event)
    



other_events_array = np.array(other_events_array)
muon_events_array = np.array(filtered_muon_events)

print("Total number of events in file:", events.num_entries)
print("Total number of two_muon_events:", len(two_muon_events))
print("Total number of two_muon_pt_cut_events:", len(two_muon_pt_cut_events))
print("Total number of muon_events_array:", len(muon_events_array))
print("Total number of filtered_other_events:", len(filtered_other_events))
print("other_events_array: ", len(other_events_array))

######################################################
# Start to construct the TLorentz Vectors (these will be the input into my model)
######################################################


print(other_events_array)



tlv = TLorentzVector()
#for evenet in muon_events_array:
 #   tlv.SetPtEtaPhiM()







