import uproot
import numpy as np
from ROOT import TLorentzVector
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#import tensorflow as tf
#from sklearn.model_selection import train_test_split

#########################################
# Opens and reads in data as numpy arrays
#########################################

file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
tree = file["Events"]
#tree.show()
data = tree.arrays(["PFCands_pdgId", 
                    "PFCands_pt",
                    "PFCands_eta",
                    "PFCands_phi",
                    "PFCands_mass",
                    "PFCands_dz",
                    ], entry_stop=500, library="np")

#########################################
#Extracts as individual arrays
#########################################
pt_array = data["PFCands_pt"]
pdgId_array = data["PFCands_pdgId"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]
dz_array = data["PFCands_dz"]
combined_pz=0.0

#########################################
# Initialize an empty list to store particles.
# This is in case I want to print values. Does no calculations
#########################################
muon_particles = np.empty((0, 4))   
non_muon_particles = np.empty((0, 4))  


#########################################
# Iterate over each event in the arrays
# and makes relevant cuts on pT and eta
#########################################

muon_pz_array = []
non_muon_pz_array=[]
labels=[]
# Initialize an empty list to store labels
labels = []

# Iterate over each event in the arrays
for event, pdgIds, pt, eta, phi, mass in zip(pt_array, pdgId_array, pt_array, eta_array, phi_array, mass_array):
    # Count the number of muons in the event
    num_muons = np.sum(abs(pdgIds == 13))

    # Check if the number of muons is exactly 2
    if num_muons == 2:
        # Initialize combined z-momentum
        combined_pz = 0.0
        
        # Iterate over each particle in the event
        for pt_val, pdgId, eta_val, phi_val, mass_val in zip(pt, pdgIds, eta, phi, mass):
            
            # Check if the particle is a muon and applies some cuts
            if abs(pdgId) == 13: 
                if abs(pt_val) > 2.0 and abs(eta_val) < 2.5: 
                    tlv_muon = TLorentzVector()
                    tlv_muon.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                    combined_pz += tlv_muon.Pz()  # Add muon's pz to combined_pz
        
        # Append combined z-momentum to labels list
        labels.append(combined_pz)

# Convert labels to numpy array
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(muon_pz_array, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=120, activation='relu'), 
    tf.keras.layers.Dense(units=60, activation='relu'), 
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=1)  
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)

# Evaluate the model on the test set
mse = model.evaluate(X_test, y_test, verbose=0)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
