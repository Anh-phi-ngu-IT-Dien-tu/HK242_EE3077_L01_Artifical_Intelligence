import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import combinations
from collections import Counter

data_path = "data.xlsx"
df = pd.read_excel(data_path)

data = df.iloc[0:10, 2:6]
dataset = data.astype(float).T.values.tolist()

table = [[row[i:i+3] for i in range(0, len(row), 3)] for row in dataset]
class_0 = [item[0] for item in table]
class_1 = [item[1] for item in table]
class_2 = [item[2] for item in table]

X = class_0 + class_1 + class_2
y = [0]*len(class_0) + [1]*len(class_1) + [2]*len(class_2)

X_array = np.array(X)
y_array = np.array(y)
classes = np.unique(y_array)

def one_vs_rest(X, y):
    models = {}
    for cls in classes:
        y_binary = (y == cls).astype(int)
        model = LogisticRegression()
        model.fit(X, y_binary)
        models[cls] = model
    return models

def predict_ovr(models, X):
    probs = np.array([model.predict_proba(X)[:, 1] for model in models.values()])
    return np.argmax(probs, axis=0)

def one_vs_one(X, y):
    models = []
    pairs = list(combinations(classes, 2))
    for cls1, cls2 in pairs:
        idx = (y == cls1) | (y == cls2)
        X_pair = X[idx]
        y_pair = y[idx]
        y_binary = (y_pair == cls1).astype(int)
        model = LogisticRegression()
        model.fit(X_pair, y_binary)
        models.append((cls1, cls2, model))
    return models

def predict_ovo(models, X):
    votes = []
    for cls1, cls2, model in models:
        pred = model.predict(X)
        vote = np.where(pred == 1, cls1, cls2)
        votes.append(vote)
    votes = np.array(votes).T
    final = [Counter(v).most_common(1)[0][0] for v in votes]
    return np.array(final)

def train_hierarchical(X, y):
    y_step1 = (y == 0).astype(int)
    model1 = LogisticRegression()
    model1.fit(X, y_step1)

    idx_1_2 = (y != 0)
    X_step2 = X[idx_1_2]
    y_step2 = y[idx_1_2]
    y_step2 = (y_step2 == 1).astype(int)
    model2 = LogisticRegression()
    model2.fit(X_step2, y_step2)

    return model1, model2

def predict_hierarchical(model1, model2, X):
    step1_pred = model1.predict(X)
    final_pred = []
    for i, pred in enumerate(step1_pred):
        if pred == 1:
            final_pred.append(0)
        else:
            pred2 = model2.predict(X[i].reshape(1, -1))[0]
            final_pred.append(1 if pred2 == 1 else 2)
    return np.array(final_pred)

ovr_models = one_vs_rest(X_array, y_array)
y_pred_ovr = predict_ovr(ovr_models, X_array)
print("One-vs-Rest Accuracy:", accuracy_score(y_array, y_pred_ovr))

ovo_models = one_vs_one(X_array, y_array)
y_pred_ovo = predict_ovo(ovo_models, X_array)
print("One-vs-One Accuracy:", accuracy_score(y_array, y_pred_ovo))

m1, m2 = train_hierarchical(X_array, y_array)
y_pred_hier = predict_hierarchical(m1, m2, X_array)
print("Hierarchical Accuracy:", accuracy_score(y_array, y_pred_hier))

#Predict with new data
# quả ổi: class 1
new_data = np.array([[6.2, 6, 5.3]])
ovr = predict_ovr(ovr_models, new_data)
ovo = predict_ovo(ovo_models, new_data)
hier = predict_hierarchical(m1, m2, new_data)
print("_"*50)
print("One-vs-Rest: Class: ", ovr[0])
print("One-vs-One: Class: ", ovo[0])
print("Hierarchical: Class: ", hier[0])