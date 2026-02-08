# Author - R Guruprasad Reddy (BLSCU4AIE24063)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def plot_knn_boundary(k_value):
    np.random.seed(1)

    x_random = np.random.randint(1, 11, 20)
    y_random = np.random.randint(1, 11, 20)

    class_labels = np.where(x_random + y_random <= 11, 0, 1)
    training_points = np.column_stack((x_random, y_random))

    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(training_points, class_labels)
    predicted_labels = knn.predict(grid_points)

    plt.scatter(grid_points[predicted_labels == 0, 0],
                grid_points[predicted_labels == 0, 1],
                color='blue', s=5)
    plt.scatter(grid_points[predicted_labels == 1, 0],
                grid_points[predicted_labels == 1, 1],
                color='red', s=5)

    plt.scatter(x_random, y_random, c=class_labels,
                edgecolor='black', s=80)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'kNN (k = {k_value})')
    print(f"Displaying plot for k={k_value}. Please close the plot window to continue...")
    plt.show()

def perform_linear_regression(csv_file_path):
    data_frame = pd.read_csv(csv_file_path)

    features = data_frame.iloc[:, 1:]
    target = data_frame.iloc[:, 0]

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)

    mse = mean_squared_error(target_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((target_test - predictions) / target_test)) * 100
    r2 = r2_score(target_test, predictions)

    return mse, rmse, mape, r2

def calculate_classification_metrics(excel_file_path):
    df = pd.read_excel(excel_file_path)

    feature_cols = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    target_col = (df['Payment (Rs)'] > df['Payment (Rs)'].median()).astype(int)

    feature_train, feature_test, target_train, target_test = train_test_split(
        feature_cols, target_col, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    feature_train = scaler.fit_transform(feature_train)
    feature_test = scaler.transform(feature_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(feature_train, target_train)

    return (
        confusion_matrix(target_train, knn.predict(feature_train)),
        confusion_matrix(target_test, knn.predict(feature_test)),
        classification_report(target_test, knn.predict(feature_test))
    )

def optimize_k_value(excel_file_path):
    df = pd.read_excel(excel_file_path)

    feature_cols = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    target_col = (df['Payment (Rs)'] > df['Payment (Rs)'].median()).astype(int)

    scaler = StandardScaler()
    feature_cols = scaler.fit_transform(feature_cols)

    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [1, 3, 5, 7, 9, 11]},
        cv=5
    )

    grid_search.fit(feature_cols, target_col)
    return grid_search.best_params_, grid_search.best_score_

def main():
    BASE = r"C:\Users\Guruprasad Reddy\Documents\Semester-4\Machine Learning"
    excel_file = BASE + r"\Lab Session Data.xlsx"
    csv_file = BASE + r"\DCT_mal.csv"

    train_cm, test_cm, report = calculate_classification_metrics(excel_file)
    print("TRAIN CONFUSION MATRIX\n", train_cm)
    print("TEST CONFUSION MATRIX\n", test_cm)
    print(report)

    mse, rmse, mape, r2 = perform_linear_regression(csv_file)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R2:", r2)

    for k in range(1, 8):
        plot_knn_boundary(k)

    best_k, score = optimize_k_value(excel_file)
    print("BEST k:", best_k)
    print("BEST CV SCORE:", score)

main()
