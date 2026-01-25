# Author - Guruprasad Reddy (bl.sc.u4aie24063)

import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
import matplotlib.pyplot as plt

# A1
def matr_rank(x):
    return np.linalg.matrix_rank(x)

def find_cost(a,b):
    return np.linalg.pinv(a)@b

# A2
def classify_cust(x):
    ans = []
    for i in x:
        if i > 200:
            ans.append("RICH")
        else:
            ans.append("POOR")
    return ans

# A3
def own_mean(x):
    x = list(x)
    l = len(x)
    return sum(x)/l

def own_var(x):
    m = own_mean(x)
    x = list(x)
    return sum((xi - m)**2 for xi in x) / len(x)

def wed_mean(x):
    return np.mean(x)

def april_mean(x):
    return np.mean(x)

def scatter_plot(x,y):
    plt.scatter(x,y)
    plt.xlabel("chg%")
    plt.ylabel("day of the week")
    plt.title("chg% vs day of the week")
    plt.show()

# A4
def explore_thyroid_data(thyroid_df):
    return {
        "data_types": thyroid_df.dtypes,
        "missing_values": thyroid_df.isnull().sum(),
        "mean": thyroid_df.mean(numeric_only=True),
        "variance": thyroid_df.var(numeric_only=True)
    }

# A5
def cal_f(data):
    f00 = 0
    f01 = 0
    f10 = 0
    f11 = 0
    for i in range(data.shape[1]):
        if data.iloc[0,i] and data.iloc[1,i] == 0:
            f00+=1
        elif data.iloc[0,i]==0 and data.iloc[1,i]==1:
            f01+=1
        elif data.iloc[0,i]==1 and data.iloc[1,i]==0:
            f10+=1
        else:
            f11+=1
    return f00,f01,f10,f11

def cal_jc_smc(f00,f01,f10,f11):
    jc = f11/(f01+f10+f11)
    smc = (f00+f11)/(f00+f01+f10+f11)
    return jc,smc

# A6
def conv_to_num(data):
    for i in range(data.shape[0]):
        if data[i,1]=="Graduation" and data[i,2]=="Single":
            data[i,1] = 1
            data[i,2] = 0
    return data

def cal_cos(data):
    return np.dot(data[0],data[1])/len(data[0])*len(data[1])

# A7
def gen_smlr_heatmap(data_matrix):
    smlr_matrix = np.corrcoef(data_matrix)
    return smlr_matrix

# A8
def impute_missing_values(data):
    for column in data.columns:
        if data[column].dtype != "object":
            data[column] = data[column].fillna(data[column].median())
        else:
            data[column] = data[column].fillna(data[column].mode()[0])
    return data

# A9
def norm_num_data(data):
    num_columns = data.select_dtypes(include=["int64", "float64"])
    data[num_columns.columns] = (
        num_columns - num_columns.min()
    ) / (num_columns.max() - num_columns.min())
    return data

def main():
    # Load Data
    data1 = pd.read_excel("Lab Session Data.xlsx",sheet_name="Purchase data")
    X = data1.iloc[:,1:4].values
    y = data1.iloc[:,4].values
    
    # Define implicitly used variables (inferred)
    data = pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    b_d = data1.iloc[:, 5:] # Binary data columns from Purchase Data

    print("--- A1 RESULTS ---")
    rank_X = matr_rank(X)
    print("The rank of the Matrix A is:", rank_X)
    
    cost = find_cost(X,y)
    print("The calculated cost vector is:\n", cost)

    print("\n--- A2 RESULTS ---")
    customer_classes = classify_cust(y)
    print("The customer classifications (showing first 5) are:", customer_classes[:5])

    print("\n--- A3 RESULTS ---")
    data2 = pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
    
    mean_val = np.mean(data2.iloc[:,3].values)
    var_val = np.var(data2.iloc[:,3].values)
    print(f"The mean of column D is {mean_val} and the variance is {var_val}")
    
    mean_own_val = own_mean(data2.iloc[:,3].values)
    print(f"The mean of column D calculated with our own function is: {mean_own_val}")
    
    var_own_val = own_var(data2.iloc[:,3].values)
    print(f"The variance of column D calculated with our own function is: {var_own_val}")

    wed_price = data2[data2["Day"]=="Wed"].iloc[:,3].values
    print(f"The mean price on Wednesdays is: {wed_mean(wed_price)}")

    april_price = data2[data2["Month"]=="Apr"].iloc[:,3].values
    print(f"The mean price in April is: {april_mean(april_price)}")

    loss_fn = lambda x : x<0
    loss = loss_fn(data2.iloc[:,8].values)
    probability = len(loss[loss == True])/len(data2.iloc[:,8])
    print(f"The probability of making a loss on the stock is: {probability}")

    profit_fn = lambda x: x>0
    profit = profit_fn(data2[data2["Day"]=="Wed"].iloc[:,8].values)
    probability_profit = len(profit[profit == True])/len(data2.iloc[:,8])
    print(f"The probability of checking a profit on Wednesdays is: {probability_profit}")

    probability_it_is_wed = len(data2[data2["Day"]=="Wed"])/len(data2.iloc[:,1])
    print(f"The conditional probability of profit given it is Wednesday is: {probability_profit/probability_it_is_wed}")

    chg_data = data2.iloc[:,8].values
    day_of_week_data = data2.iloc[:,2].values
    print("Generating scatter plot for Change % vs Day of Week...")
    scatter_plot(chg_data, day_of_week_data)

    print("\n--- A4 RESULTS ---")
    print("Exploring thyroid data properties:")
    exploration_results = explore_thyroid_data(data)
    print(f"Data Types:\n{exploration_results['data_types']}")
    print(f"Missing Values Count:\n{exploration_results['missing_values']}")
    # Skipping heavy print of mean/var to keep it clean, or just printing specific parts
    
    print("\n--- A5 RESULTS ---")
    # Handling potential NaN in b_d for logic or filling 0
    # Ensuring b_d is used as expected by cal_f
    f00,f01,f10,f11 = cal_f(b_d.fillna(0)) 
    print(f"Calculated f00: {f00}, f01: {f01}, f10: {f10}, f11: {f11}")
    
    jc,smc = cal_jc_smc(f00,f01,f10,f11)
    print(f"The Jaccard Coefficient is: {jc}")
    print(f"The Simple Matching Coefficient is: {smc}")

    print("\n--- A6 RESULTS ---")
    # conv_to_num modifies data in place usually or returns it. check function.
    # conv_to_num(data) uses data[i,1], so data must be numpy array or access like one.
    # but 'data' is a dataframe. this might fail if not .values or if implicit.
    # Using data.values for A6 as per common pattern
    try:
        converted_data = conv_to_num(data.values) 
        cosine_sim = cal_cos(converted_data)
        print(f"The Cosine Similarity is: {cosine_sim}")
    except Exception as e:
        print(f"Could not calculate cosine similarity due to data format compatibility: {e}")

    print("\n--- A7 RESULTS ---")
    similarity_matrix = gen_smlr_heatmap(X)
    print("The generated Similarity Matrix is:\n", similarity_matrix)

    print("\n--- A8 RESULTS ---")
    imputed_data = impute_missing_values(data.copy())
    print("Missing values have been imputed successfully.")

    print("\n--- A9 RESULTS ---")
    normalized_data = norm_num_data(imputed_data)
    print("Numeric data has been normalized successfully.")

if __name__ == "__main__":
    main()