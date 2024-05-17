import uproot
import numpy as np
from ROOT import TLorentzVector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

###########################################
# Open and read data from the ROOT file
# /nfs/home... path is for isaac access.
# root://eospublic path is for local access.

###########################################
#file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
file = uproot.open("/nfs/home/avendras/nano_data2016_1.root")
tree = file["Events"]
data = tree.arrays(["PFCands_pdgId", 
                    "PFCands_pt", 
                    "PFCands_eta",
                    "PFCands_phi", 
                    "PFCands_mass"], entry_stop=200500, library="np")


###########################################
# Extracts the data as individual arrays
###########################################
pdgId_array = data["PFCands_pdgId"]
pt_array = data["PFCands_pt"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]


###########################################
# Initialize lists for different types of particles
###########################################
muon_pz_list = []
non_muon_pz_list = []


###########################################
# Process each event based on 
# teh Z->mumu decay process.
###########################################
for pdgIds, pts, etas, phis, masses in zip(pdgId_array, pt_array, eta_array, phi_array, mass_array):
    muon_pz = 0
    non_muon_pz = 0
    muon_count = 0
    for pdgId, pt, eta, phi, mass in zip(pdgIds, pts, etas, phis, masses):
        tlv = TLorentzVector()
        tlv.SetPtEtaPhiM(pt, eta, phi, mass)
        if abs(pdgId) == 13 and pt > 2.0 and abs(eta) < 2.5:
            muon_pz += tlv.Pz()
            muon_count += 1
        else:
            non_muon_pz += tlv.Pz()
    if muon_count == 2:
        muon_pz_list.append(muon_pz)
        non_muon_pz_list.append(non_muon_pz)


###########################################
# Converting the lists of PFCands combined Pz 
# into numpy arrays.
###########################################
combined_muon_pz_array = np.array(muon_pz_list)
combined_non_muon_pz_array = np.array(non_muon_pz_list)


###########################################
# Pads the smaller array(muons) with zeros
# to match the length of the larger array(non_muons)
###########################################
size_difference = len(combined_muon_pz_array) - len(combined_non_muon_pz_array)
if size_difference > 0:
    combined_non_muon_pz_array = np.pad(combined_non_muon_pz_array, (0, size_difference), mode='constant', constant_values=0)
    

###########################################
# Splits the non_muon PFCand dataset 
# and uses the muon PFCand dataset as labels
###########################################
X_train, X_test, y_train, y_test = train_test_split(combined_non_muon_pz_array, combined_muon_pz_array, test_size=0.2)


###########################################
# Defines a standard DNN
###########################################
architecture_details = "10-100-10-1"
X_train = X_train.reshape(-1, 1)  # Reshape to (num_samples, 1)
X_test = X_test.reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])


###########################################
# Compiles, trains and evaluates the model
###########################################
learning_rate = 0.00001
model.compile(optimizer =
        tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="mean_absolute_error")
history = model.fit(X_train, y_train, epochs=30, batch_size=300, verbose=2, validation_data=(X_test, y_test))
mae = model.evaluate(X_test, y_test, verbose=0)
print("Mean Absolute Error:", mae)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
y_pred = model.predict(X_test)


###########################################
# Visualize Predictions and losses of the model
###########################################
plt.figure()
plt.scatter(y_test, y_pred,s=1,color='orange')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs. Predicted Labels')
plt.savefig("prediction" + architecture_details + ".png")
plt.show()

plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Learning Curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig("learing_curve_and_loss" + architecture_details + ".png")
plt.show()


