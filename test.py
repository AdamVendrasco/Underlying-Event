#import ROOT
import uproot
#from ROOT import RDataFrame
#import pandas as pd
#import awkard as ak
#import TensorFlow as tf
import numpy
import sys
#'root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root'

nano_file= uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
#keys = nano_file.GetlistOfKeys()
#for key in keys:
#   obj = key.ReadObj()             # Read the object corresponding to the key
#   if isinstance(obj, ROOT.TTree): # Check if the object is a TTree
#           print(obj.GetName())

# Define your list of cut conditions
cut_conditions = [
    "PFCands_pdgId = 13",  # Select events 
    #"abs(Muon_eta) < 2.4",  # Select 
    #"muon_charge == -1"  # Select eveevents = nano_file["Events"]

]
numpy.set_printoptions(threshold=sys.maxsize)
#combined_cut_condition = " & ".join(cut_conditions)
events = nano_file["Events"]
pf_id = events["PFCands_pdgId"].array(library = "np")
pf_phi = events["PFCands_phi"].array(library = "np")
pf_eta = events["PFCands_eta"].array(library = "np")
pf_m = events["PFCands_mass"].arrays(library = "np")


print(pf_id.tolist())
