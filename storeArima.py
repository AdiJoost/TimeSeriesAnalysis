import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
import utils
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

csvSaveName = "arimaStatistics.csv"
failedCSVNames = "failedCSV.csv"

def main():
    createCSV()
    pairs = utils.loadStoreDepartmentPairs("longUniqueTimeSeries.csv")
    df = pd.read_csv("data/train.csv")
    for pair in tqdm(pairs):
        try:
            analyse(pair, df)
        except Exception as e:
            saveFailesCSV(pair[0], pair[1])

def createCSV():
    with open(csvSaveName, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId", "ma", "ar", "nDiff", "shapiroStat", "ShapiroValue", "ljungStat", "ljungValue"])
    with open(failedCSVNames, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId"])

def analyse(pair, df):
    storeTrain, storeTest = utils.getStoreDF(pair[0], pair[1], df, 0)
    diff, n = utils.getStationaryValue(storeTrain)
    ma = max(utils.getACF(diff, nlags=52, confidence=0.3))
    ar = max(utils.getPACF(diff, nlags=52, confidence=0.3))
    model = utils.getArimaFit(diff, ar=ar, nDiff=n, ma=ma)
    residuals = model.resid
    shapiro_test_statistic, p_value = shapiro(residuals)
    ljung_box_test = acorr_ljungbox(residuals, lags=[ar], return_df=True)
    ljung_box_stat = ljung_box_test.iloc[0,0]
    ljung_box_value = ljung_box_test.iloc[0,1]
    saveCSV (pair[0], pair[1], ma, ar, n, shapiro_test_statistic, p_value, ljung_box_stat, ljung_box_value)

def saveCSV(storeId, departmentId, ma, ar, nDiff, shapiroTestStatistic, shapiroPValue, ljungStat, ljungPvalue):
    with open(csvSaveName, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId, ma, ar, nDiff, shapiroTestStatistic, shapiroPValue, ljungStat, ljungPvalue ])

def saveFailesCSV(storeId, departmentId):
    with open(failedCSVNames, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId])

if __name__ == "__main__":
    main()


