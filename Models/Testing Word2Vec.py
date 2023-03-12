from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

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
import gensim


# load ascii text and covert to lowercase
filename = "/Users/michael/Documents/Programming 3/Science Fair 2020/Models/wonderland.txt"
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

AllSplits = []
for z in range(len(rawpeptides)):
    splitup = []
    for k in range(len(rawpeptides[z])):
        splitup.append(rawpeptides[z][k])
    AllSplits.append(splitup)
W2Vmodel = gensim.models.Word2Vec(AllSplits, min_count=10)


seq_length = 15
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = actualtext[i:i + seq_length]
    seq_out = actualtext[i + seq_length]
    space = 0
    for z in range(len(seq_in)):
        if seq_in[z] == ' ':
            space = 1
    if space == 0:
        dataX.append(seq_in)
        dataY.append(real_char_to_int[seq_out])

empty = np.zeros((dataX.__len__(), seq_length, 100))

for z in range(empty.shape[0]):
    print(z)
    for f in range(len(dataX[0])):
        empty[z, f, :] = W2Vmodel[dataX[z][f]]
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]

X = empty
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(150, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.4))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=100, batch_size=128, callbacks=callbacks_list)

# Generation
# load the network weights
filename = "weights-improvement-100-2.0184.hdf5"
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


