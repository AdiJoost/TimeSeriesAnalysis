import pandas as pd
import utils
import numpy as np
import csv
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

RESULTS_CSV = "arimaXStatistics.csv"
STORE_BETTER_CSV = "/Users/karolina/Git/time_series/TimeSeriesAnalysis/fuel_price/results/arimaxEvaluationBetter.csv"
STORE_WORSE_CSV = "/Users/karolina/Git/time_series/TimeSeriesAnalysis/fuel_price/results/arimaxEvaluationWorse.csv"
STORE_FAILED = "/Users/karolina/Git/time_series/TimeSeriesAnalysis/fuel_price/results/arimaxEvaluationFailed.csv"
NUMBER_OF_TEST_POINTS = 12


def main():
    better_count = 0
    worse_count = 0
    failed_count = 0

    for file in [RESULTS_CSV, "data/train.csv", "data/features.csv"]:
        if not os.path.exists(file):
            print(f"Error: Missing required file {file}")
            return

    createStoreCSV()

    results = pd.read_csv(RESULTS_CSV)
    df = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")
    features["Date"] = pd.to_datetime(features["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = pd.merge(df, features[["Date", "Store", "Fuel_Price"]], on=["Date", "Store"], how="left")

    for _, row in tqdm(results.iterrows(), total=len(results)):
        try:
            evaluate(df, row["storeId"], row["departmentId"], row["nDiff"], row["ar"], row["ma"])
            if isBetter(row["storeId"], row["departmentId"]): 
                better_count += 1
            else:
                worse_count += 1
        except Exception as e:
            print(f"Error processing Store ID: {row['storeId']}, Department ID: {row['departmentId']}: {e}")
            saveFailed(row["storeId"], row["departmentId"])
            failed_count += 1

    print("\nEvaluation Summary:")
    print(f"Better evaluations: {better_count}")
    print(f"Worse evaluations: {worse_count}")
    print(f"Failed evaluations: {failed_count}")

def isBetter(storeId, departmentId):
    with open(STORE_BETTER_CSV, "r", encoding="utf-8") as file:
        return any(
            str(storeId) in row and str(departmentId) in row
            for row in csv.reader(file)
        )


def createStoreCSV():
    with open(STORE_BETTER_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId", "ma", "ar", "nDiff", "baseLineMAE", "actualMAE"])
    with open(STORE_WORSE_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId", "ma", "ar", "nDiff", "baseLineMAE", "actualMAE"])
    with open(STORE_FAILED, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId"])

def evaluate(df, storeId, departmentId, nDiff, ar, ma):
    try:
        storeTrain, storeTest = utils.getStoreDF(storeId, departmentId, df, NUMBER_OF_TEST_POINTS)
        
        if storeTrain is None or storeTest is None or len(storeTrain) == 0 or len(storeTest) == 0:
            print(f"Missing or insufficient data for Store ID: {storeId}, Department ID: {departmentId}")
            saveFailed(storeId, departmentId)
            return

        currentSeries = storeTrain["Weekly_Sales"]
        differentials = []
        for i in range(int(nDiff)):
            currentSeries = currentSeries.diff().dropna()
            differentials.append(currentSeries.copy())

        aligned_fuel_price = storeTrain['Fuel_Price'].loc[currentSeries.index]
        model = ARIMA(currentSeries, exog=aligned_fuel_price, order=(ar, nDiff, ma))
        fitted = model.fit()

        lastDate = storeTrain["Date"].max()
        arimax_forecast = utils.getForecast(fitted, differentials, lastDate, NUMBER_OF_TEST_POINTS, exog=storeTest['Fuel_Price'])
        baseLineMAE = getBaseLine(storeTrain["Weekly_Sales"], storeTest["Weekly_Sales"])
        actualMAE = getMAE(arimax_forecast["Weekly_Sales"], storeTest["Weekly_Sales"])

        if actualMAE < baseLineMAE:
            saveValueBetter(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE)
        else:
            saveValueWorse(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE)

    except Exception as e:
        print(f"Error for Store ID: {storeId}, Department ID: {departmentId}: {e}")
        saveFailed(storeId, departmentId)

def getBaseLine(train, test):
    baseLine = getBaseLineForecast(train, NUMBER_OF_TEST_POINTS)
    return getMAE(baseLine, test)

def getBaseLineForecast(train, numberOfPredictions):
    return train[-52: -52+numberOfPredictions]

def getMAE(forecast, actual):
    sum = 0
    for prediction, actualValue in zip(forecast, actual):
        sum += np.abs(actualValue - prediction)
    return sum / len(actual)

def saveValueBetter(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE):
    with open(STORE_BETTER_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId, ma, ar, nDiff, baseLineMAE, actualMAE])

def saveValueWorse(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE):
    with open(STORE_WORSE_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId, ma, ar, nDiff, baseLineMAE, actualMAE])

def saveFailed(storeId, departmentId):
    with open(STORE_FAILED, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId])

if __name__ == "__main__":
    main()