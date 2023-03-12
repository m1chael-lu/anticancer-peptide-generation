import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras import regularizers
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

from keras.models import Sequential


from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import keras.backend as K
from sklearn.metrics import f1_score
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import PReLU
import matplotlib.image as mpimg
from sklearn.feature_selection import SelectFromModel


chosen = 150


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def Scores(predictions, actual):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(actual.shape[0]):
        if actual[i, 0] == 1:
            if predictions[i, 0] ==1:
                tn += 1
        if actual[i, 1] == 1:
            if predictions[i, 1] == 1:
                tp += 1
        if actual[i, 0] == 1:
            if predictions[i, 1] == 1:
                fp += 1
        if actual[i, 1] == 1:
            if predictions[i, 0] == 1:
                fn += 1
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    precision = tp/(tp + fp)
    return sensitivity, specificity, precision

DFx = pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/Physico DataX.csv")
DFy = pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/Physico DataY.csv")

Xdata = np.array(DFx)
Ydata = np.array(DFy)

Xdata = Xdata[:, 1:]
Ydata = Ydata[:, 1]

# indexes = np.arange(len(Ydata))
# np.random.shuffle(indexes)

NormalizedX = np.zeros(Xdata.shape)
# Normalizing data
for z in range(Xdata.shape[1]):
    Min = np.min(Xdata[:, z])
    Max = np.max(Xdata[:, z])
    if Max-Min == 0:
        NormalizedX[:, z] = Xdata[:, z]
    else:
        for k in range(Xdata.shape[0]):
            NormalizedX[k, z] = (Xdata[k, z] - Min)/(Max-Min)


selector = SelectKBest(chi2, k=chosen).fit(NormalizedX,Ydata)
scores = selector.scores_
scores = np.nan_to_num(scores)
ranked_scores = scores.argsort()[-chosen:][::-1]
sorted_scores = np.sort(scores)

X_new = SelectKBest(chi2, k=chosen).fit_transform(NormalizedX, Ydata)
# X_new = NormalizedX

split = int(0.6 * Xdata.shape[0])

X_train = X_new[:split, :]
X_test = X_new[split:, :]
Y_train = np_utils.to_categorical(Ydata[:split], 2)
Y_test = np_utils.to_categorical(Ydata[split:], 2)

# model = RandomForestClassifier(n_estimators=400, max_depth=3)
# model.fit(X_train, Y_train)
#
# featureImportance = model.feature_importances_
# selector = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=chosen)
#
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

# START
iterations = 5
TrainScores = []
TestScores = []
for i in range(iterations):
    model = Sequential()

    model.add(Dense(50, input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(200, kernel_regularizer=regularizers.l2(0.001)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(25, kernel_regularizer=regularizers.l2(0.001)))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=["accuracy"])
    model.fit(X_train, Y_train, epochs=75, verbose=1, validation_split=0.2, batch_size=32)


    y_pred = model.predict(X_train)
    y_empty = np.zeros((len(y_pred), 2))
    idx_max = np.argmax(y_pred,1)
    for x in range(len(y_pred)):
        y_empty[x, idx_max[x]] = 1

    score = 0
    for z in range(y_empty.shape[0]):
        if y_empty[z, 0] == Y_train[z, 0]:
            score += 1
    TrainAccuracy = score/y_empty.shape[0]

    y_pred = model.predict(X_test)
    y_empty = np.zeros((len(y_pred), 2))
    idx_max = np.argmax(y_pred,1)
    for x in range(len(y_pred)):
        y_empty[x, idx_max[x]] = 1

    score = 0
    for z in range(y_empty.shape[0]):
        if y_empty[z, 0] == Y_test[z, 0]:
            score += 1
    TestAccuracy = score/y_empty.shape[0]

    print("Training Accuracy: " + str(TrainAccuracy))
    print("Testing Accuracy: " + str(TestAccuracy))
    TrainScores.append(TrainAccuracy)
    TestScores.append(TestAccuracy)
print("Average Training Accuracy: " + str(np.average(TrainScores)))
print("Average Testing Accuracy: " + str(np.average(TestScores)))


#
sensitivity, specificity, precision = Scores(y_empty, Y_test)
print("Sensitivity: "+ str(sensitivity))
print("Specificity: "+ str(specificity))
# print("Precision: "+ str(precision))

# F1 score

Y_true = Ydata[split:]
Y_pred = model.predict(X_test)
idx_max = np.argmax(Y_pred, 1)
f1_score(Y_true, idx_max)

