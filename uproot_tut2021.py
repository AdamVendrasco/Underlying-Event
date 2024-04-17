import awkward as ak
import uproot

#'root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root'

nano_file= uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
events = nano_file["Events"]
pfcans = events.arrays(filter_name="PF*",entry_stop = 10,library="pd")
print(pfcans)


pfcsv = pfcans.to_csv('pfcan_csv')
