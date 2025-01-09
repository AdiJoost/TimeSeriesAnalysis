import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def getStoreDF(storeId: int, departementId: int, testSize: int=0):
    df = pd.read_csv("data/train.csv")
    store = df[(df["Store"] == storeId) & (df["Dept"] == departementId)]
    if (len(store) < 1):
        return None, None
    store['Date'] = pd.to_datetime(store['Date'], format='%Y-%m-%d')
    if (testSize > 0):
        store_train = store[:-testSize]
        store_test = store[-testSize:]
        return store_train, store_test
    return store, None

def getStoreDF(storeId: int, departementId: int, dataFrame, testSize: int=0):
    store = dataFrame[(dataFrame["Store"] == storeId) & (dataFrame["Dept"] == departementId)]
    if (len(store) < 1):
        return None, None
    store['Date'] = pd.to_datetime(store['Date'], format='%Y-%m-%d')
    if (testSize > 0):
        store_train = store[:-testSize]
        store_test = store[-testSize:]
        return store_train, store_test
    return store, None

def loadStoreDepartmentPairs(filepath):
     uniqueIds = pd.read_csv(filepath)
     return [(row["storeId"], row["departmentId"]) for _, row in uniqueIds.iterrows()]
     

def getStationaryValue(store, maxNumberOfRepetitions: int=3, pValue:int=0.05):
    series = store["Weekly_Sales"]
    numberOfDifferentiation = 0
    while(numberOfDifferentiation < maxNumberOfRepetitions and adfuller(series)[1] > pValue):
        series = series.diff().dropna()
        numberOfDifferentiation += 1
    return series, numberOfDifferentiation

def plot_ACF_PACF(stationary, lags=30):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(stationary, lags=lags, ax=ax[0], title='ACF of Differenced Data')
    plot_pacf(stationary, lags=lags, ax=ax[1], title='PACF of Differenced Data')
    plt.show()

def getACF(stationary, pValue=0.05, confidence=0.25, nlags=20):
    acfValues, acfConfidence = acf(stationary, nlags=nlags, alpha=pValue, fft=False)
    return [i for i, acfValue in enumerate(acfValues) if abs(acfValue) > confidence]

def getPACF(stationary, pValue=0.05, confidence=0.25, nlags=20):
    pacfValues, pacfConfidence = pacf(stationary, nlags=nlags, alpha=pValue)
    return [i for i, pacfValues in enumerate(pacfValues) if abs(pacfValues) > confidence]

def getArimaFit(stationary, ar, nDiff, ma):
    model = ARIMA(stationary, order=(ar,nDiff, ma))
    return model.fit()

def getForecast(model, diffs, lastDate, predictionLength, exog=None):
    newDates = pd.date_range(start=lastDate + pd.Timedelta(weeks=1), periods=predictionLength, freq="W-FRI")
    
    if exog is not None:
        forecast_values = model.forecast(steps=predictionLength, exog=exog)
    else:
        forecast_values = model.forecast(steps=predictionLength)
    
    forecast = pd.DataFrame({"Date": newDates, "Weekly_Sales": forecast_values})

    for diff in diffs:
        forecast.iloc[0, 1] = diff.iloc[-1] + forecast.iloc[0, 1]
        for i in range(1, predictionLength):
            forecast.iloc[i, 1] = forecast.iloc[i, 1] + forecast.iloc[i - 1, 1]
    
    return forecast


def plotForeCast(store_train, store_test, forecast):
    plt.figure(figsize=(12, 6))
    plt.plot(store_train['Date'], store_train['Weekly_Sales'], label='Historical Sales')
    plt.plot(forecast["Date"], forecast["Weekly_Sales"], label='Forecasted Sales', color='red')
    plt.plot(store_test["Date"], store_test["Weekly_Sales"], label='Actual Sales', color='green')
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def isNormalDistributedShapiro(residuals, pValueThreshold = 0.05):
    shapiro_test_statistic, p_value = shapiro(residuals)
    return p_value > pValueThreshold

def noAutoCorrelationInResidual(residuals, lag, pValueThreshold=0.05):
    ljung_box_test = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    return ljung_box_test.iloc[0,1] > pValueThreshold