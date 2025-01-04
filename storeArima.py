import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
import utils
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

csvSaveName = "arimaStatistics.csv"

def main():
    createCSV()
    pairs = utils.loadStoreDepartmentPairs("longUniqueTimeSeries.csv")
    df = pd.read_csv("data/train.csv")
    for i in tqdm(range(5)):
        analyse(pairs[i], df)

def createCSV():
    with open(csvSaveName, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["storeId", "departmentId", "ma", "ar", "nDiff", "shapiroStat", "ShapiroValue", "ljungStat", "ljungValue"])

def analyse(pair, df):
    storeTrain, storeTest = utils.getStoreDF(pair[0], pair[1], df, 0)
    diff, n = utils.getStationaryValue(storeTrain)
    ma = max(utils.getACF(diff, nlags=60, confidence=0.25))
    ar = max(utils.getPACF(diff, nlags=60, confidence=0.25))
    model = utils.getArimaFit(diff, ar=ar, nDiff=n, ma=ma)
    PreviousWeekly = storeTrain["Weekly_Sales"]
    lastDate = storeTrain["Date"].max()
    forecast = utils.getForecast(model, (PreviousWeekly,), lastDate, 12)
    residuals = model.resid
    shapiro_test_statistic, p_value = shapiro(residuals)
    ljung_box_test = acorr_ljungbox(residuals, lags=[ar], return_df=True)
    ljung_box_stat = ljung_box_test.iloc[0,0]
    ljung_box_value = ljung_box_test.iloc[0,1]
    saveCSV (pair[0], pair[1], ma, ar, nDiff, shapiro_test_statistic, p_value, ljung_box_stat, ljung_box_value)

def saveCSV(storeId, departmentId, ma, ar, nDiff, shapiroTestStatistic, shapiroPValue, ljungStat, ljungPvalue):
    with open(csvSaveName, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([storeId, departmentId, ma, ar, nDiff, shapiroTestStatistic, shapiroPValue, ljungStat, ljungPvalue ])

if __name__ == "__main__":
    main()


