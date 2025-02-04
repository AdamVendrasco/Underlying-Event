import uproot
import numpy as np
from ROOT import TLorentzVector
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#########################################
# Opens and reads in data as numpy arrays
#########################################
file = uproot.open("/nfs/home/avendras/Underlying-Event/nano_data2016_1.root")
tree = file["Events"]
#tree.show()
data = tree.arrays(["PFCands_pdgId",
                    "PFCands_pt",
                    "PFCands_eta",
                    "PFCands_phi",
                    "PFCands_mass",
                    "PFCands_dz",
                    ], entry_stop=100, library="np")
#########################################
# Extracts as individual arrays
#########################################
pt_array = data["PFCands_pt"]
pdgId_array = data["PFCands_pdgId"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]
dz_array = data["PFCands_dz"]
combined_pz = 0.0
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
non_muon_pz_array = []
labels = []
# Iterate over each event in the arrays
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
                    tlv_muon = TLorentzVector()
                    tlv_muon.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                    muon_pz_array.append(tlv_muon.Pz())  # Append the muon momentum component to the list
                    combined_pz += tlv_muon.Pz()  # Add muon's pz to combined_pz
                    labels.append(combined_pz)
            # checks it is NOT a muon and does the same just no cuts
            if abs(pdgId) != 13:
                tlv_non_muon = TLorentzVector()
                tlv_non_muon.SetPtEtaPhiM(pt_val, eta_val, phi_val, mass_val)
                pz_non_muon = tlv_non_muon.Pz()
                non_muon_particles = np.append(non_muon_particles, [[pt_val, pdgId, eta_val, phi_val]], axis=0)
                non_muon_pz_array.append(pz_non_muon)
non_muon_pz_array = np.array(non_muon_pz_array)
labels = np.array(labels)
print(len(muon_pz_array))
#########################################
# Start of the Model creation and training
#########################################
X_train, X_test, y_train, y_test = train_test_split(muon_pz_array, labels, test_size=0.2, random_state=42)

X_train = np.array(muon_pz_array).reshape(-1, 1)
X_test = np.array(muon_pz_array).reshape(-1, 1)
y_train = np.array(labels)
y_test = np.array(labels)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1000, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=550, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=200, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=1)
])
# Compile the model with custom linear regression loss function
model.compile(optimizer='adam', loss="mean_squared_error")
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=2)
# Evaluate the model on the test set
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error on Test Set:", mse)

# Visualize Predictions vs. Ground Truth
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs. Predicted Labels')
#plt.show()
plt.savefig("prediction.png")

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig("model_loss.png")
