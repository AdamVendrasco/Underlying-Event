import uproot
import awkward as ak
import pprint

# Open the ROOT file
data_file = "/nfs/home/avendras/Underlying-Event/root_files/002DAE91-77A7-E511-B61B-00266CFAEA48.root"
file = uproot.open(data_file)

# Access the tree
tree = file["Events"]

# Define the branch name
branch_name =  "recoTracks_generalTracks__RECO./recoTracks_generalTracks__RECO.obj/recoTracks_generalTracks__RECO.obj.hitPattern_.hitPattern[50]"

# Extract the branch data
#branch_data = tree[branch_name].array()
branch_data = ak.to_list(tree[branch_name].array())
# Print the first 5 events' data for clarity
for i, event in enumerate(branch_data[:2]):
    print(f"Event {i+1}: {event}")
