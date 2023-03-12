import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
from Bio.Seq import Seq
from Bio import Alphabet
from Bio.Alphabet import Reduced
from keras.utils import np_utils
import sys


# load ascii text and covert to lowercase
filename = "/Users/michael/Documents/programming/Science Fair 2020/ALL ACPs.csv"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

rawpeptides = np.array(pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/ALL ACPs.csv"))[:, 1]
text = ''

reduced_peptides = []

for i in range(rawpeptides.size):
	sequence = rawpeptides[i]
	seq = ''
	for z in range(len(sequence)):
		seq += Alphabet.Reduced.murphy_10_tab[sequence[z]]
	reduced_peptides.append(seq)
	text += reduced_peptides[i]
	text += ' '

actualtext = ''

for i in range(rawpeptides.size):
	actualtext += rawpeptides[i]
	actualtext += ' '

# create mapping of unique chars to integers
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

real_chars = sorted(list(set(actualtext)))
real_char_to_int = dict((c, i) for i, c in enumerate(real_chars))

# summarize the loaded data
n_chars = len(text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 15
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = text[i:i + seq_length]
	seq_out = actualtext[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(real_char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np_utils.to_categorical(dataX, len(chars))

# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=25, batch_size=128, callbacks=callbacks_list)



# Generation
# load the network weights
filename = "weights-improvement-25-1.0487.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))
real_int_to_char = dict((i, c) for i, c in enumerate(real_chars))

# pick a random seed

# Ensuring that the seed starts with a blank
while True:
	start = np.random.randint(0, len(dataX)-1)
	if dataX[start][0] == 0:
		break

pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

generatedCodes = actualtext[start:start+seq_length]

# generate characters
for i in range(300):
	oldpattern = pattern
	x = np_utils.to_categorical(pattern, len(chars)).reshape((1, seq_length, 11))
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	generatedCodes += real_int_to_char[index]

	if real_int_to_char[index] != ' ':
		reduced = char_to_int[Alphabet.Reduced.murphy_10_tab[real_int_to_char[index]]]
	else:
		reduced = 0

	oldpattern.append(reduced)
	oldpattern = oldpattern[1:len(pattern)]
	pattern = oldpattern
print(generatedCodes)


