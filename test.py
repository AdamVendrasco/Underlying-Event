#import ROOT
import uproot
import numpy as np
import sys
#'root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root'

nano_file= uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")

events = nano_file["Events"]
pf_id = events["PFCands_pdgId"].array(library = "np")

# Define a function to check if an event contains exactly two muons
def contains_two_muons(event):
    muon_count = np.sum(np.abs(event) == 13)                                        #Count the number of muons (pdgId = 13)
    return muon_count == 2


events_with_two_muons = [event for event in pf_id if contains_two_muons(event)]     #Filter the pf_id array to select events with two muons

event_array=[]                                                                                       
for idx, event in enumerate(events_with_two_muons):                                 #Print the filtered events
    #print(f"Event {idx + 1}: {event}")
    event_array.append(events_with_two_muons)

print("Total number of events in file: ",events.num_entries)
print("Total number of events with exactly 2 muons: ", len(events_with_two_muons))   
    

