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
from keras.utils import np_utils

def Scores(predictions, actual):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(actual.size):
        if actual[i] == 0:
            if predictions[i] == 1:
                fp += 1
        if actual[i] == 1:
            if predictions[i] == 0:
                fn += 1
        if actual[i] == 1:
            if predictions[i] == 1:
                tp += 1
        if actual[i] == 0:
            if predictions[i] == 0:
                tn += 1
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

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)

#
# # 5 fold Cross Validation
#
# C_optimized_start = 10000000.0
# C_steps = 10
# kernelOptimized = 'linear'
# kernel_steps = 4
#
# CVarray = np.zeros((C_steps, kernel_steps))
#
# for x in range(kernel_steps):
#     for y in range(C_steps):
#         print(y)
#         if x == 1:
#             kernelOptimized = 'poly'
#         elif x == 2:
#             kernelOptimized = 'sigmoid'
#         elif x == 3:
#             kernelOptimized = "rbf"
#         C_optimized = C_optimized_start / (10**y)
#         CVmodel = svm.SVC(kernel=kernelOptimized, C=C_optimized)
#         CVarray[y, x] = np.mean(cross_val_score(CVmodel, X_train, Y_train, cv=5))
#
# degree_start = 1
# degree_steps = 6
# Selectedfeautures = 200
# feature_steps = 12
# stepsize = 25
# C_optimized_start = 10000000.0
# C_steps = 10
#
# CVarray = np.zeros((degree_steps, feature_steps, C_steps))
#
# for x in range(degree_steps):
#     print("degree: " + str(x))
#     for y in range(feature_steps):
#         print(y)
#         for z in range(C_steps):
#             print ("C values: " + str(z))
#             degree_Optimized = degree_start + x
#             Featurecount = Selectedfeautures + y * stepsize
#
#             X_new = SelectKBest(chi2, k=Featurecount).fit_transform(NormalizedX, Ydata)
#             split = int(0.6 * Xdata.shape[0])
#
#             X_train = X_new[:split, :]
#             Y_train = Ydata[:split]
#
#             C_optimized = C_optimized_start / (10 ** z)
#             CVmodel = svm.SVC(kernel="poly", C=C_optimized, degree = degree_Optimized)
#             CVarray[x, y, z] = np.mean(cross_val_score(CVmodel, X_train, Y_train, cv=5, scoring='f1_macro'))
#
#
# pd.DataFrame(CVarray).to_csv("/Users/michael/Documents/Programming 3/Science Fair 2020/Saved Data/Cross validation 2.csv")
# pd.DataFrame(CVarray).to_csv("/Users/michael/Documents/Programming 3/Science Fair 2020/Saved Data/Cross validation 1.csv")
# pd.DataFrame(CVarray).to_csv("/Users/michael/Documents/Programming 3/Science Fair 2020/Saved Data/Cross validation ALL.csv")

# Final Evaluation

iterations = 5
TrainScores = []
TestScores = []

for i in range(iterations):
    # model = svm.SVC(kernel="rbf", C=80, probability=True)
    # model = svm.SVC(kernel="rbf", C=2, probability=True)
    # model = svm.SVC(kernel="poly", degree=3)
    # model.fit(X_train, Y_train)

    # model = KNeighborsClassifier(n_neighbors=30, algorithm="brute", leaf_size=10)
    # model.fit(X_train, Y_train)
    #
    model = RandomForestClassifier(n_estimators=300, max_depth=3)
    model.fit(X_train, Y_train)

    training_accuracy=model.score(X_train, Y_train)
    testing_accuracy = model.score(X_test, Y_test)

    proba = model.predict(X_test)
    sensitivity, specificity, precision = Scores(proba, Y_test)
    print(sensitivity, specificity, precision)

    print("Training Accuracy: " + str(training_accuracy))
    print("Testing Accuracy: " + str(testing_accuracy))
    TrainScores.append(training_accuracy)
    TestScores.append(testing_accuracy)

print("Average Training Accuracy: " + str(np.average(training_accuracy)))
print("Average Testing Accuracy: " + str(np.average(testing_accuracy)))
