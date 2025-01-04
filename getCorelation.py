from statsmodels.tsa.stattools import ccf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import utils
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PAIRS_PATH = "twoYearTimeSeries.csv"
SIGNIFICANT_CSV = "significant.csv"
INSIGNIFICANT_CSV = "insignificant.csv"

def main():
    pairs = utils.loadStoreDepartmentPairs(PAIRS_PATH)
    createSignificantCSV()
    createNotSignificantCSV()
    data  = pd.read_csv("data/train.csv")
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    features = pd.read_csv("data/features.csv")
    features['Date'] = pd.to_datetime(features['Date'], format='%Y-%m-%d')
    for pair in tqdm(pairs):
        store = data[(data["Store"] == pair[0]) & (data["Dept"] == pair[1])]
        pValue = getPvalue(store, features)
        if pValue < 0.05:
            addSignificant(pair)
        else:
            addInsignificant(pair)


def createSignificantCSV():
    with open(SIGNIFICANT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId"])

def createNotSignificantCSV():
    with open(INSIGNIFICANT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId"])

def getPvalue(store, features):
    meged = pd.merge(store, features[["Date", "Fuel_Price"]], on="Date", how="left")
    cross_corr = ccf(meged['Weekly_Sales'], meged['Fuel_Price'])
    X = meged['Weekly_Sales']
    y = meged['Fuel_Price']
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X).fit()
    return model.f_pvalue

def addSignificant(pair):
    with open(SIGNIFICANT_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([pair[0], pair[1]])

def addInsignificant(pair):
    with open(INSIGNIFICANT_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([pair[0], pair[1]])

if __name__ == "__main__":
    main()