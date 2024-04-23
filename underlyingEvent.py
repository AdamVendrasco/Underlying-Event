:q#import ROOT
import uproot
#from ROOT import RDataFrame
import numpy as np
#import pandas as pd
#import awkard as ak
#import TensorFlow as tf


#'root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root'

nano_file= uproot.open("root://eospublic.cern.ch//eos/opendata/cms/derived-data/PFNano/29-Feb-24/SingleMuon/Run2016G-UL2016_MiniAODv2_PFNanoAODv1/240207_205649/0000/nano_data2016_1.root")
#keys = nano_file.GetlistOfKeys()
#for key in keys:
#	obj = key.ReadObj()  			# Read the object corresponding to the key
#	if isinstance(obj, ROOT.TTree):	# Check if the object is a TTree
#			print(obj.GetName())

events = nano_file["Events"]
pf_dz = events.arrays(filter_name="PFCands_dz",library="np")
pf_dz_csv = pf_dz.to_csv('pf_dz_csv')




# Define your list of cut conditions
#cut_conditions = [
#    "muon_pt > 20",  # Select events where muon_pt is greater than 20 GeV
#    #"abs(muon_eta) < 2.4",  # Select events where the absolute value of muon_eta is less than 2.4
    #"muon_charge == -1"  # Select events where muon_charge is equal to -1
#]

# Combine the cut conditions using logical AND (all conditions must be satisfied)
#combined_cut_condition = " & ".join(cut_conditions)

# Apply the combined cut and retrieve the selected data
#selected_data = tree.arrays("*", cut=combined_cut_condition)

# Now you can access the data of the selected events and process it as needed





#Overall plan
#Step 1: Combine the root files.
		#Currently I am only running over 1 file

#Step 2: What data is relevant in the root files?
		#Find out what data is important (tracks, pt..etc.)

#Step 3: Convert data from root TTree to arrays
		#Figure out what kind of arrays to use. (Pandas, numpy, awkard)
		#Currently above this is using pandas
		#How will keras/tensor flow handel this?

# Step 4: Define Model Architecture
		#Pretty sure I dont need a "complicated" model. Perhaps just a ordinary Deep NN.
		#I do not think I need a classifcation NN or Convolution NN. Talk with larry to be sure but perhaps something like this,

#model = Sequential([
#    Dense(64, activation='relu', input_shape=(input_shape,)),  # Adjust input_shape based on your data
#    Dense(64, activation='relu'),
#	Dense(1, activation='sigmoid')
#])
#	complie the model
#model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#	Train the model
#model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
#
#	Evaluate Model
#loss, accuracy = model.evaluate(X_test, y_test)
#print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")




