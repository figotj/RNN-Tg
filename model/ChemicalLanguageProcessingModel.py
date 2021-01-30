#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding, Bidirectional, TimeDistributed, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, r2_score


def getRNNmodel(LSTMunits, out_dim, input_dim):

	RNNmodel = Sequential()
	RNNmodel.add(Embedding(input_dim, out_dim, input_length=input_len))
	RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
	RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
	RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
	RNNmodel.add(Reshape((int(LSTMunits/2*input_len),)))
	RNNmodel.add(Dense(1))

	return RNNmodel

##############################################################################################

# import the dataset
polymer_df = pd.read_csv("../data/polymer_tg.txt", delim_whitespace=True)
Tg = polymer_df["Tg-celsius"]
Tg = np.asarray(Tg)

# import the rep of polymers
tokenized_polymer = np.load("../data/Tokened_polymer_int120.npy")

#
input_dim = 45 # the number of unique tokens: including space
input_len = 120 # the max len of the array
out_dim_arr = np.asarray([5,10,15,20,25])
LSTM_units_arr = np.asarray([10,20,30,40,50,60])

X_train, X_test, y_train, y_test = train_test_split(tokenized_polymer, Tg, test_size=0.1,
                                                    random_state=42)

# save the test dataset
np.save("Polymer_test.npy", X_test)
np.save("Tg_test.npy", y_test)

# grid search
for out_dim in out_dim_arr:
	for LSTMunits in LSTM_units_arr:

		RNNmodel = getRNNmodel(LSTMunits, out_dim, input_dim)

		RNNmodel.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])

		# fit model
		# add callbacks
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
		model_name = "best_model_" + str(LSTMunits) + "_" + str(out_dim) + ".h5"
		mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=0, save_best_only=True)

		history = RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=500, \
								 batch_size=128, callbacks=[es, mc])

		# save the history
		history_df = pd.DataFrame.from_dict(history.history)
		name = "history_" + str(LSTMunits) + "_" + str(out_dim) + ".csv"
		history_df.to_csv(name, index=True)
