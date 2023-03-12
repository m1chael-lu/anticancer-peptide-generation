import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score


DFx = pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/Physico DataX.csv")
DFy = pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/Physico DataY.csv")

Xdata = np.array(DFx)
Ydata = np.array(DFy)

Xdata = Xdata[:, 1:]
Ydata = Ydata[:, 1]

MaxNorm = np.zeros((Xdata.shape[1]))
MinNorm = np.zeros((Xdata.shape[1]))

NormalizedX = np.zeros(Xdata.shape)
# Normalizing data
for z in range(Xdata.shape[1]):
    Min = np.min(Xdata[:, z])
    Max = np.max(Xdata[:, z])
    MaxNorm[z] = Max
    MinNorm[z] = Min
    if Max-Min == 0:
        NormalizedX[:, z] = Xdata[:, z]
    else:
        for k in range(Xdata.shape[0]):
            NormalizedX[k, z] = (Xdata[k, z] - Min)/(Max-Min)

GenerationRaw = np.array(pd.read_csv("/Users/michael/Documents/programming/Science Fair 2020/RNNgeneration.csv"))[:, 1:]
NormalizedGen = np.zeros(GenerationRaw.shape)
for z in range(GenerationRaw.shape[1]):
    Max = MaxNorm[z]
    Min = MinNorm[z]
    if Max-Min == 0:
        NormalizedGen[:, z] = GenerationRaw[:, z]
    else:
        for k in range(GenerationRaw.shape[0]):
            NormalizedGen[k, z] = (GenerationRaw[k, z] - Min)/(Max-Min)



chosen = 150



# Chi squared feature selection
# selector = SelectKBest(chi2, k=chosen).fit(NormalizedX,Ydata)
# scores = selector.scores_
# scores = np.nan_to_num(scores)
# ranked_scores = scores.argsort()[-chosen:][::-1]
# sorted_scores = np.sort(scores)
#
# X_new = SelectKBest(chi2, k=chosen).fit_transform(NormalizedX, Ydata)

split = int(0.6 * Xdata.shape[0])

X_train = NormalizedX[:split, :]
X_test = NormalizedX[split:, :]
Y_train = Ydata[:split]
Y_test = Ydata[split:]


# Decision Tree feature selection

model = RandomForestClassifier(n_estimators=400, max_depth=3)
model.fit(X_train, Y_train)

featureImportance = model.feature_importances_
selector = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=chosen)

selected_indexes = selector.get_support()

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)


SelectedGen = np.zeros((NormalizedGen.shape[0], chosen))
for k in range(NormalizedGen.shape[0]):
    temp = NormalizedGen[k, :]
    SelectedGen[k, :] = temp[selected_indexes]

# Final Evaluation

# iterations = 25
# TrainScores = []
# TestScores = []

# model = svm.SVC(kernel="rbf", C=80, probability=True)
# model = svm.SVC(kernel="rbf", C=2, probability=True)
model = svm.SVC(kernel="poly", degree=3, probability=True)
# model.fit(X_train, Y_train)

# model = KNeighborsClassifier(n_neighbors=30, algorithm="brute", leaf_size=10)
# model.fit(X_train, Y_train)
#
# model = RandomForestClassifier(n_estimators=300, max_depth=3)
model.fit(X_train, Y_train)

training_accuracy=model.score(X_train, Y_train)
testing_accuracy = model.score(X_test, Y_test)

# Saving Model
filename = 'SVMModelRBF.sav'
pickle.dump(model, open(filename, 'wb'))

# Loading Model
model = pickle.load(open(filename, 'rb'))


proba = model.predict_proba(SelectedGen)
Positive = proba[:, 1]


Threshold = np.zeros(proba.shape[0])
for k in range(proba.shape[0]):
    if proba[k, 1] > 0.95:
        Threshold[k] = 1

countvalues = [324, 553, 820, 1144, 1392]
true_probs = proba[0, 1] + proba[countvalues[0], 1] + proba[countvalues[1], 1] + proba[countvalues[2], 1] + proba[countvalues[3], 1]

pd.DataFrame(Positive).to_csv("/Users/michael/Documents/Programming 3/Science Fair 2020/Saved Data/RNN generation accuracies.csv")
pd.DataFrame(Threshold).to_csv("/Users/michael/Documents/programming/Science Fair 2020/Generated Data/ThresholdGen1.csv")


# Prepare Mathew's Correlation Coefficient

matthews_corrcoef(Y_test, model.predict(X_test))
f1_score(Y_test, model.predict(X_test))