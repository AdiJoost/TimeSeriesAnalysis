import pandas as pd
import utils
import numpy as np
import csv
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

RESULTS_CSV = "results/arima10Statistics.csv"
STORE_BETTER_CSV = "results/arima10EvaluationBetter.csv"
STORE_WORSE_CSV = "results/arima10EvaluationWorse.csv"
STORE_FAILED = "results/arima10EvaluationFailed.csv"
NUMBER_OF_TEST_POINTS = 12

def main():
    createStoreCSV()
    results = pd.read_csv(RESULTS_CSV)
    df = pd.read_csv("data/train.csv")
    for _, row in tqdm(results.iterrows()):
        try:
            evaluate(df, row["storeId"], row["departmentId"], row["nDiff"], row["ar"], row["ma"])
        except Exception as E:
            saveFailed(row["storeId"], row["departmentId"])

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
    storeTrain, storeTest = utils.getStoreDF(storeId, departmentId, df, NUMBER_OF_TEST_POINTS)
    currentSeries = storeTrain["Weekly_Sales"]
    differentials = []
    for i in range(int(nDiff)):
        currentSeries = currentSeries.diff().dropna()
        differentials.append(currentSeries.copy())

    model = ARIMA(currentSeries, order=(ar, nDiff, ma))
    fitted = model.fit()
    lastDate = storeTrain["Date"].max()
    forecast = utils.getForecast(fitted, differentials, lastDate, NUMBER_OF_TEST_POINTS)
    baseLineMAE = getBaseLine(storeTrain["Weekly_Sales"], storeTest["Weekly_Sales"])
    actualMAE = getMAE(forecast["Weekly_Sales"], storeTest["Weekly_Sales"])
    if (actualMAE < baseLineMAE):
        saveValueBetter(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE)
    else:
        saveValueWorse(storeId, departmentId, nDiff, ar, ma, baseLineMAE, actualMAE)

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