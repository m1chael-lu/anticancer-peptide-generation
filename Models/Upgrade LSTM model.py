import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
from Bio.Seq import Seq
from Bio import Alphabet
from Bio.Alphabet import Reduced
from keras.utils import np_utils
import sys

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
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 15
dataX = []
dataY = []

startingLocations = [0]

for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = actualtext[i + seq_length]
    spaces = 0
    for z in range(len(seq_in)):
        if seq_in[z] == ' ':
            spaces += 1
    if seq_in[0] == ' ':
        startingLocations.append(i+1)
    if spaces == 0:
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(real_char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

npX = np.array(dataX)

# reshape X to be [samples, time steps, features]
X = np_utils.to_categorical(npX, len(chars))
split =.8
splitMark = int(np.round(split*n_patterns))

X_train = X[:splitMark]
X_test = X[splitMark:]
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

Y_train = y[:splitMark]
Y_test = y[splitMark:]
# define the LSTM model
model = Sequential()
model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.50))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X_train, Y_train, epochs=110, batch_size=64, callbacks=callbacks_list)


result = model.predict(X_test)
choice = np.argmax(result, axis=1)
true = np.argmax(Y_test, axis=1)

score = 0
for i in range(len(choice)):
    if choice[i]==true[i]:
        score += 1
test_accuracy = score/len(choice)

result = model.predict(X_train)
choice = np.argmax(result, axis=1)
true = np.argmax(Y_train, axis=1)

score = 0
for i in range(len(choice)):
    if choice[i]==true[i]:
        score += 1
train_accuracy = score/len(choice)

# Generation
# load the network weights
filename = "weights-improvement-108-0.8018.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))
real_int_to_char = dict((i, c) for i, c in enumerate(real_chars))

# pick a random seed

# Ensuring that the seed starts with a blank

generatedSequences = []
for x in range(10):
    start = np.random.randint(0, len(startingLocations))
    X_index = startingLocations[start] - 15 * start
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    generatedCodes = actualtext[start:start + seq_length]
    count = 0
    while True:

    # generate characters
        oldpattern = pattern
        x = np_utils.to_categorical(pattern, len(chars)).reshape((1, seq_length, 11))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        if real_int_to_char[index] == ' ':
            generatedSequences.append(generatedCodes)
            break
        generatedCodes += real_int_to_char[index]
        if real_int_to_char[index] != ' ':
            reduced = char_to_int[Alphabet.Reduced.murphy_10_tab[real_int_to_char[index]]]
        else:
            oldpattern.append(reduced)
            oldpattern = oldpattern[1:len(pattern)]
            pattern = oldpattern
        count += 1
        if count == 50:
            break
print(generatedCodes)
