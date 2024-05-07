import uproot
import numpy as np
from ROOT import TLorentzVector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Open and read data from the ROOT file
file = uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
#file = uproot.open("/nfs/home/avendras/nano_data2016_1.root")
tree = file["Events"]
data = tree.arrays(["PFCands_pdgId", 
                    "PFCands_pt", 
                    "PFCands_eta",
                    "PFCands_phi", 
                    "PFCands_mass"], entry_stop=50000, library="np")

# Extract data as individual numpy arrays
pdgId_array = data["PFCands_pdgId"]
pt_array = data["PFCands_pt"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]

# Initialize lists for different types of particles
muon_pz_list = []
non_muon_pz_list = []

# Process each event
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

# Convert lists to numpy arrays
combined_muon_pz_array = np.array(muon_pz_list)
combined_non_muon_pz_array = np.array(non_muon_pz_list)

print(combined_muon_pz_array.shape)
print(combined_non_muon_pz_array.shape)

# Pad the smaller array with zeros to match the length of the larger array
size_difference = len(combined_muon_pz_array) - len(combined_non_muon_pz_array)
if size_difference > 0:
    combined_non_muon_pz_array = np.pad(combined_non_muon_pz_array, (0, size_difference), mode='constant', constant_values=0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(combined_non_muon_pz_array, combined_muon_pz_array, test_size=0.2, random_state=42)

# Define and compile a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu', input_shape=(1,)),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(500, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(130, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile the model with custom linear regression loss function
model.compile(optimizer='adam', loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=2)

# Evaluate the model on the test set
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error on Test Set:", mse)

train_loss = history.history['loss']
y_pred = model.predict(X_test)

# Visualize Predictions of the model
plt.figure()
plt.scatter(y_test, y_pred,s=1)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs. Predicted Labels')
plt.savefig("test_git_prediction.png")

# Visualize loss function per epochs
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('test_git_model_losses.png')
