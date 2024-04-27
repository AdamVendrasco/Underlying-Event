import uproot
import numpy as np
from ROOT import TLorentzVector

# Open the NanoAOD file
file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
tree = file["Events"]
branches = tree.arrays()

tree.arrays(["PFCands_pt","PFCands_phi"])
#double_muon_mask = branches['nMuon'] == 2
#pt_mask = branches['nMuon'] >2.0


#cuts= branches['PFCands_pt'][double_muon_mask][pt_mask]
#print(cuts)

#print(len(cuts))