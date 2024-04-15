import uproot	
import awkward as ak
import numpy as np
import ROOT
from ROOT import RDataFrame

print("check1")
data= uproot.open('root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root')

print("check2")

print(data.keys())
print(data["Events"])

tree=data["Events"]
print("check3")
print("blah")
print("check4")
PFCands_dz = tree["PFCands_dz"].array(library="ak")[:1]
print("check5")
print(PFCands_pt)
print("check6")
