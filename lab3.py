import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import math

def dot_product(a,b):
    dot = 0
    for i in range(len(a)):
        dot += a[i]*b[i]
    return dot
    
def euc_norm(x):
    s = 0
    for i in range(len(x)):
        s += x[i]**2
    return math.sqrt(s)

def mean(x):
    return sum(x)/len(x)

def var(x):
    m = mean(x)
    return sum((xi - m) ** 2 for xi in x) / len(x)

def std_dev(x):
    return math.sqrt(var(x))

def matrix(m):
    means = []
    std_devs = []
    for i in range(m.shape[1]):
        col = m[:,i]
        means.append(mean(col))
        std_devs.append(std_dev(col))
    return np.array(means),np.array(std_devs)

def den_pattern(x):
    df = pd.read_excel("DCT_mal.xlsx")
    feature = df[x].dropna()
    mean = np.mean(feature)
    var = np.var(feature)
    print(f"Mean of {x}: {mean}")
    print(f"Variance of {x}: {var}")

    plt.figure()
    plt.hist(feature,bins=10)
    plt.title(f"Histogram of {x}")
    plt.xlabel(f"{x} Values")
    plt.ylabel("Frequency")
    plt.show()

def minkowski_dist(x,y,p):
    d = 0
    for i in range(len(x)):
        d += abs(x[i] - y[i]) ** p
    return d**(1/p)

def knn_class(X_train, y_train, X_test, k=3, p=2):
    pred = []
    for x in X_test:
        dist = []
        for i in range(len(X_train)):
            d = minkowski_dist(x, X_train[i], p)
            dist.append((d, y_train[i]))
        dist.sort(key=lambda x: x[0])
        neighbors = dist[:k]
        labels = [label for _, label in neighbors]
        prediction = max(set(labels), key=labels.count)
        pred.append(prediction)
    return np.array(pred)

def confusion_mat(yt, yp):
    labels = np.unique(yt)
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_index = {label: i for i, label in enumerate(labels)}
    for i in range(len(yt)):
        t = label_index[yt[i]]
        p = label_index[yp[i]]
        cm[t][p] += 1
    return cm, labels

def acc(cm):
    return np.trace(cm) / np.sum(cm)

def prec(cm):
    n = cm.shape[0]
    precisions = []
    for i in range(n):
        tp = cm[i][i]
        fp = np.sum(cm[:,i]) - tp
        if tp + fp == 0:
            precisions.append(0)
        else:
            precisions.append(tp / (tp + fp))
    return np.mean(precisions)

def rec(cm):
    n = cm.shape[0]
    recalls = []
    for i in range(n):
        tp = cm[i][i]
        fn = np.sum(cm[i]) - tp
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(tp / (tp + fn))
    return np.mean(recalls)

def f1(cm):
    n = cm.shape[0]
    f = []
    for i in range(n):
        tp = cm[i][i]
        fp = np.sum(cm[:,i]) - tp
        fn = np.sum(cm[i]) - tp
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f.append(f1)
    return np.mean(f1)

def main():

    A = np.array([1, 2, 3])
    B = np.array([5, 6, 7])

    print("Dot Product :", dot_product(A, B))
    print("Dot Product(using numpy) :", np.dot(A, B))

    print("Euclidean Norm  of A :", euc_norm(A))
    print("Euclidean Norm (using numpy) of A :", np.linalg.norm(A))

    print("Euclidean Norm of B :", euc_norm(B))
    print("Euclidean Norm of B (using numpy) :", np.linalg.norm(B))

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "DCT_mal.csv")

    df = pd.read_csv(file_path)

    classes = df["LABEL"].unique()
    class1 = df[df["LABEL"] == classes[0]]
    class2 = df[df["LABEL"] == classes[1]]

    features = [col for col in df.columns if col != "LABEL"]

    X1 = class1[features].values
    X2 = class2[features].values

    c1, s1 = matrix(X1)
    c2, s2 = matrix(X2)

    print("Class 1 Centroid :", c1)
    print("Class 2 Centroid :", c2)
    print("Class 1 Spread :", s1)
    print("Class 2 Spread :", s2)

    print("Interclass Distance between centroids:",
          np.linalg.norm(c1 - c2))

    col_name = features[0]
    feature_data = df[col_name]

    print(f"Mean of {col_name}:", feature_data.mean())
    print(f"Variance of {col_name}:", feature_data.var())

    plt.hist(feature_data, bins=10)
    plt.title(f"Histogram of {col_name}")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    f14 = df.loc[0, features].values
    f24 = df.loc[1, features].values

    distances = []

    for p in range(1, 11):
        distances.append(minkowski_dist(f14, f24, p))

    for i in range(1, 11):
        print(f"minkowski distance {i} : {distances[i-1]}")

    plt.figure()
    plt.plot(range(1, 11), distances)
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    for p in range(1, 11):
        scipy_dist = minkowski(f14, f24, p)
        print(f"minkowski distance (using scipy) {p} : {scipy_dist}")

    X6 = df[features].values
    Y6 = df["LABEL"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X6, Y6, test_size=0.2, stratify=Y6, random_state=42
    )

    print("Total samples:", len(X6))
    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
    print("Training labels size:", y_train.shape)
    print("Testing labels size:", y_test.shape)

    n = KNeighborsClassifier(n_neighbors=3)
    n.fit(X_train, y_train)

    y = n.predict(X_test)
    print("Prediction :", y)
    print("Actual labels:", y_test)

    acc_builtin = n.score(X_test, y_test)
    print("Accuracy (built in function) :", acc_builtin)

    y_pred = knn_class(X_train, y_train, X_test, k=3)
    accown = np.mean(y_pred == y_test)
    print("Accuracy (own function):", accown)

    accuracies = []
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acck = knn.score(X_test, y_test)
        accuracies.append(acck)
        print(f"k = {k}, Accuracy = {acck}")

    print("Accuracy for k = 1 : ", accuracies[0])
    print("Accuracy for k = 3 : ", accuracies[2])

    plt.figure()
    plt.plot(range(1, 11), accuracies, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k for kNN Classifier")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(10)
    plt.close()

    ytrain = n.predict(X_train)
    ytest = n.predict(X_test)

    print("Confusion Matrix (Training Data):")
    print(confusion_matrix(y_train, ytrain))

    print("Confusion Matrix (Testing Data):")
    print(confusion_matrix(y_test, ytest))

    print("Training :")
    print("Precision:", precision_score(y_train, ytrain, average='weighted'))
    print("Recall   :", recall_score(y_train, ytrain, average='weighted'))
    print("F1-score :", f1_score(y_train, ytrain, average='weighted'))

    print("Testing :")
    print("Precision:", precision_score(y_test, ytest, average='weighted'))
    print("Recall   :", recall_score(y_test, ytest, average='weighted'))
    print("F1-score :", f1_score(y_test, ytest, average='weighted'))

    conmat, labels = confusion_mat(y_test, y_pred)

    print("Confusion Matrix:")
    print(conmat)
    print("Labels:", labels)

    print("Precision:", prec(conmat))
    print("Recall   :", rec(conmat))
    print("F1-score :", f1(conmat))

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    Y_train_oh = np.eye(len(np.unique(y_train_enc)))[y_train_enc]

    W = np.linalg.pinv(X_train) @ Y_train_oh

    scores = X_test @ W
    y_pred_mat = np.argmax(scores, axis=1)
    y_pred_mat = le.inverse_transform(y_pred_mat)

    acc_mat = accuracy_score(y_test, y_pred_mat)

    print("Matrix Inversion Accuracy:", acc_mat)
    
main()