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

chosen = 2049
#
selector = SelectKBest(chi2, k=chosen).fit(NormalizedX,Ydata)
scores = selector.scores_
scores = np.nan_to_num(scores)
ranked_scores = scores.argsort()[-chosen:][::-1]
sorted_scores = np.sort(scores)

X_new = SelectKBest(chi2, k=chosen).fit_transform(NormalizedX, Ydata)

split = int(0.6 * Xdata.shape[0])

X_train = X_new[:split, :]
X_test = X_new[split:, :]
Y_train = Ydata[:split]
Y_test = Ydata[split:]

features = np.arange(2049)


model = RandomForestClassifier(n_estimators=400, max_depth=3)
model.fit(X_train, Y_train)

featureImportance = model.feature_importances_
selector = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=250)


