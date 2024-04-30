import uproot
import numpy as np
from ROOT import TLorentzVector
#import tensorflow as tf
#from sklearn.model_selection import train_test_split


#########################################
# Opens and reads in data as numpy arrays
#########################################
file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
tree = file["Events"]
tree.show()
data = tree.arrays(["PFCands_pdgId", 
                    "PFCands_pt",
                    "PFCands_eta",
                    "PFCands_phi",
                    "PFCands_mass",
                    "PFCands_dz",
                    ], entry_stop=50, library="np")

#########################################
#Extracts as individual arrays
#########################################
pt_array = data["PFCands_pt"]
pdgId_array = data["PFCands_pdgId"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]
dz_array = data["PFCands_dz"]


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

for event, pdgIds, pt, eta, phi, mass in zip(pt_array, pdgId_array, pt_array, eta_array, phi_array, mass_array):
    # Count the number of muons in the event
    num_muons = np.sum(abs(pdgIds == 13))

    # Check if the number of muons is exactly 2
    if num_muons == 2:
        # Iterate over each particle in the event
        for pt_val, pdgId, eta_val, phi_val, mass_val in zip(pt, pdgIds, eta, phi, mass):
            
            # Check if the particle is a muon and applies some cuts
            if abs(pdgId) == 13: 
                if abs(pt_val) > 2.0 and abs(eta_val) < 2.5: 

                    #Contstructs TLorentz vector for muons
                    tlv_muon = TLorentzVector()
                    tlv_muon.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                    #tlv_muon.SetPz(pt_val * np.sinh(eta_val))
                    

                    muon_particles = np.append(muon_particles, [[pt_val, pdgId, eta_val, phi_val]], axis=0)
            
            # checks it is NOT a muon and does the same just no cuts 
            if abs(pdgId) != 13:
                tlv_non_muon = TLorentzVector()
                tlv_non_muon.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
            
                #tlv_non_muon.SetPz(pt_val * np.sinh(eta_val))
                pz_non_muon = tlv_non_muon.Pz()
                non_muon_particles = np.append(non_muon_particles, [[pt_val, pdgId, eta_val, phi_val]], axis=0)

#for particle in muon_particles:
#    print("Muon eta : ",particle[2])
#    print("Muon pt : ", particle[0])
#
#for particle in non_muon_particles:
#    print("Non Muon eta : ",particle[2])
#    print("Non Muon pt : ", particle[0])


#########################################
# Start of the Model creation and tranining
#########################################

# Define custom loss function
#def custom_loss(y_true, y_pred):
#    # Define custom logic for loss calculation
#    custom_loss_value = tf.reduce_mean(tf.abs(y_true - y_pred))  # Example: Mean Absolute Error
#    return custom_loss_value

#X = []  # List to store z-momentum of non-muon particles
#y = []  # List to store z-momentum of Z-boson particles
#
#for tlv_non_muon in non_muon_particles:
#    # Extract z-momentum
#    pz_non_muon = tlv_non_muon.Pz()
#    X.append([pz_non_muon])  # Append z-momentum to input features
#    # Assuming you have the z-momentum of Z-boson particles stored somewhere
#    y.append(z_boson_pz)  # Append z-momentum of Z-boson as target
#
## Converts lists to numpy arrays
#X = np.array(X)
#y = np.array(y)
#
## Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
## Build the Model
#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]),  # Input layer
#    tf.keras.layers.Dense(units=32, activation='relu'),  # Hidden layer 1
#    tf.keras.layers.Dense(units=16, activation='relu'),  # Hidden layer 2
#    tf.keras.layers.Dense(units=1)  # Output layer
#])
#
## Compiles the Model
#model.compile(optimizer='adam', loss='custom_loss_value')
#
## Train the Model
#model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


