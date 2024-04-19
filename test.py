import uproot

# Open the ROOT file
file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")

# Access the TTree containing event information
tree = file["tree_name"]  # Replace "tree_name" with the actual name of your TTree

# Get the total number of events in the TTree
num_events = tree.num_entries

# Loop over each event and number them
for event_number in range(num_events):
    # Print or process the event number (starting from 0)
    print(f"Event {event_number}:")
    # Access data for this event if needed
    data = tree.arrays("*", entry_start=event_number, entry_stop=event_number + 1)
    # Process the data as needed
