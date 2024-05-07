import uproot
import numpy as np
from ROOT import TLorentzVector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Open and read data from the ROOT file
file = uproot.open("/nfs/home/avendras/nano_data2016_1.root")
entry_stop=15000
tree = file["Events"]
data = tree.arrays(["PFCands_pdgId", "PFCands_pt", "PFCands_eta", "PFCands_phi", "PFCands_mass"],
        entry_stop=entry_stop, library="np")

# Extract data as individual numpy arrays
pdgId_array = data["PFCands_pdgId"]
pt_array = data["PFCands_pt"]
eta_array = data["PFCands_eta"]
phi_array = data["PFCands_phi"]
mass_array = data["PFCands_mass"]

# Initialize lists for different types of particles
combined_muon_pz_array = []
combined_non_muon_pz_array = []

# Process each event
for pdgIds, pts, etas, phis, masses in zip(pdgId_array, pt_array, eta_array, phi_array, mass_array):
    muon_count = 0
    non_muon_pz_array = []  # Initialize non-muon pz array for each event
    for pdgId, pt, eta, phi, mass in zip(pdgIds, pts, etas, phis, masses):
        if abs(pdgId) == 13 and pt > 2.0 and abs(eta) < 2.5:
            muon_count += 1
        else:
            tlv_non_muon = TLorentzVector()
            tlv_non_muon.SetPtEtaPhiM(pt, eta, phi, mass)
            non_muon_pz_array.append(tlv_non_muon.Pz())  # Append non-muon pz to the array
    if muon_count == 2:
        for pdgId, pt, eta, phi, mass in zip(pdgIds, pts, etas, phis, masses):
            if abs(pdgId) == 13:
                tlv_muon = TLorentzVector()
                tlv_muon.SetPtEtaPhiM(pt, eta, phi, mass)
                combined_muon_pz_array.append(tlv_muon.Pz())
        # Extend the combined non-muon pz array with the current event's non-muon pz array
        combined_non_muon_pz_array.extend(non_muon_pz_array)

# Convert lists to numpy arrays
combined_muon_pz_array = np.array(combined_muon_pz_array)
combined_non_muon_pz_array = np.array(combined_non_muon_pz_array)

# Ensure both arrays have the same number of samples
min_samples = min(len(combined_muon_pz_array), len(combined_non_muon_pz_array))
combined_muon_pz_array = combined_muon_pz_array[:min_samples]
combined_non_muon_pz_array = combined_non_muon_pz_array[:min_samples]

print(combined_muon_pz_array.shape)
print(combined_non_muon_pz_array.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(combined_non_muon_pz_array, combined_muon_pz_array, test_size=0.2, random_state=42)

# Define and compile a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, activation='relu', input_shape=(1,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.BatchNormalization(),
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
plt.scatter(y_test, y_pred, s=1)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('True vs. Predicted Labels')
plt.show()
plt.savefig("pred.png")

# Visualize loss function per epochs
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

