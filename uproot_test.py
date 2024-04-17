import uproot	
import awkward as ak
import numpy as np
import ROOT
from ROOT import RDataFrame
import pandas
import awkward-pandas

print("check1")
data= uproot.open('root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root')
print("check2")

print(data.keys())
print(data["Events"])

tree=data["Events"]
print("check3")
tree.show()

#PFCands_dz = tree["PFCands_dz"].array(library="ak")[:1]
print("check4")

#print(PFCands_pt)
print("check5")


#Look into using pandas?#Currently getting runtime errors
PFak = tree.arrays(filter_name = "PFCands_mass",library="ak")
print(PFak)
PFpd=tree.arrays(filter_name= "PFCands_mass",library="pd")
print(PFpd)
