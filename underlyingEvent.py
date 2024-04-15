import ROOT
import uproot
from ROOT import RDataFrame
import numpy as np
from root_pandas import read_root


dataFile = ROOT.TFile.Open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")

keys = dataFile.GetListOfKeys()
for key in keys:
	obj = key.ReadObj()  # Read the object corresponding to the key
	if isinstance(obj, ROOT.TTree):  # Check if the object is a TTree
			print(obj.GetName())

print("checkpoint 1")

df= read_root("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root","Events")

#flat_tree = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")['Events']

print("checkpoint 2")




#print(flat_tree.keys())

print("checkpoint 4")
