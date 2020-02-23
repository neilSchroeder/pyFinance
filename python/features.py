import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from collections import Counter
from sklearn import svm, neighbors, model_selection
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_tickers():
    df = pd.read_csv(('data/sp500_joinedClose.csv'), index_col = 0)
    return df.columns.values.tolist()

def get_XDayPctChange(ticker, days):
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, days+1):
        df['{}_{}DayPctChange'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return df

def get_MA(ticker, days):
    weights = np.repeat(1., days)/days
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayMA'.format(ticker,days)] = z
    df.fillna(0,inplace=True)

    return df

def get_MA(ticker, df, days):
    weights = np.repeat(1., days)/days
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayMA'.format(ticker,days)] = z
    df.fillna(0,inplace=True)

    return df

def get_StdDev(ticker, df, days, nDeviations):
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    zp = []
    zm = []
    for i,val in enumerate(df[ticker]):
        if i > int(days):
            zp.append(nDeviations*np.std(df[ticker][i-days:i+1]))
            zm.append(-1 * nDeviations*np.std(df[ticker][i-days:i+1]))
        else:
            zp.append(0.)
            zm.append(0.)

    df['{}_{}Day_{}StdDevs'.format(ticker,days,nDeviations)] = np.array(zp)
    df['{}_{}Day_{}StdDevs'.format(ticker,days,-1*nDeviations)] = np.array(zm)
    df.fillna(0,inplace=True)

    return df

def get_EMA(ticker, days):
    weights = np.exp(np.linspace(-1.,0., days))
    weights /= weights.sum()
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayEMA'.format(ticker,days)] = z

    return df

def get_EMAfromTicker(ticker, df, days):
    weights = np.exp(np.linspace(-1.,0., days))
    weights /= weights.sum()
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayEMA'.format(ticker,days)] = z

    return df

def get_MDI(ticker, days):
    ret = []
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()

    for i,close in enumerate(df[ticker]):
        if i == 0:
            ret.append(close)
        else:
            if ret[-1] != 0:
                mdi = ret[-1] + (close - ret[-1])/0.6/days/((close/ret[-1])**4)
                ret.append(mdi)
            else:
                ret.append(close)

    df['{}_{}DayMDI'.format(ticker,days)] = np.array(ret)
    return df

def get_MDIfromTicker(ticker, df, days):
    ret = []
    tickers = df.columns.values.tolist()
    for i,close in enumerate(df[ticker]):
        print(i, close)
        if i == 0:
            ret.append(close)
        else:
            print(close, ret[-1])
            if ret[-1] != 0 and close != 0 and not np.isinf(ret[-1]):
                mdi = ret[-1] + (close - ret[-1])/0.6/days/((close/ret[-1])**4)
                ret.append(mdi)
            else:
                ret.append(close)

    df['{}_{}DayMDI'.format(ticker,days)] = np.array(ret)

    return df

""" Exctracts the XDayPctChange indicator and prepares it for ml.
    Currently set up for a 1 week (7 Day) predictor """

def getFeature_XDayPctChange(ticker, days):
    df = get_XDayPctChange(ticker, days)

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df

""" Extracts the MACD indicator and prepares it for ml.
    MACD is defined to be the 12 Day EMA - 26 Day EMA
    The trigger for MACD is the point at which it crosses the 9 Day MACD EMA
"""

def getFeature_MACD(ticker):
    df12 = get_EMA(ticker, 12)
    df = get_EMA(ticker, 26)

    df['{}_macd'.format(ticker)] = df12['{}_12DayEMA'.format(ticker)] - df['{}_26DayEMA'.format(ticker)]
    df_signalMacd = get_EMAfromTicker('{}_macd'.format(ticker), df, 9)

    df['{}_target1'.format(ticker)] =  df['{}_macd'.format(ticker)] - df_signalMacd['{}_macd_9DayEMA'.format(ticker)]

    return df


def get_AdvancesAndDeclines(ticker):
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    advances = []
    declines = []
    last_rows = []
    for date, row in df.iterrows():
        last_rows.append(row)
        advances.append(0)
        declines.append(0)
        for zicker in tickers:
            if len(last_rows) > 1:
                if row[zicker] - last_rows[-2][zicker] > 0:
                    advances[-1] += 1
                elif row[zicker] - last_rows[-2][zicker] < 0:
                    declines[-1] += 1

    df['{}_advances'.format(ticker)] = np.array(advances)
    df['{}_declines'.format(ticker)] = np.array(declines)
    zm = np.array(advances)-np.array(declines)
    zp = np.array(advances)+np.array(declines)
    df['{}_advMinusDec'.format(ticker)] = np.array(advances) - np.array(declines)
    df['{}_adjNetAdvances'.format(ticker)] = np.divide(zm,zp)

    return df

""" Calculates the McClellan Oscillator or adjusted McClellan Oscillator
    for the S&P500. This oscillator is uses EMAs of the Adjusted Net Advances.
    The oscillator is defined to be the difference of the 19 Day EMA of ANA and
    the 39 Day EMA of ANA
"""

def getFeature_McClellanOscillator(ticker):
    """get the advances and declines"""
    df = get_AdvancesAndDeclines(ticker)
    df = get_EMAfromTicker('{}_adjNetAdvances'.format(ticker), df, 19)
    df = get_EMAfromTicker('{}_adjNetAdvances'.format(ticker), df, 39)
    df['{}_McClellan'.format(ticker)] = df['{}_adjNetAdvances_19DayEMA'.format(ticker)] - df['{}_adjNetAdvances_39DayEMA'.format(ticker)]

    return df

""" Calculates the MACD using the McGinley Dynamic Indicator.
    The rule of thumb seems to be use half the period of an EMA to get similar
    performance. So, there is a 6 day MDI and a 13 day MDI. Signaling is done on
    the difference.
"""

def getFeature_MDICD(ticker):
    df6 = get_MDI(ticker, 6)
    df = get_MDI(ticker, 13)
    df['{}_mdicd'.format(ticker)] = df6['{}_6DayMDI'.format(ticker)] - df['{}_13DayMDI'.format(ticker)]

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    return df

def getFeature_BollingerBands(ticker):
    df = pd.read_csv('data/sp500_typicalPrice.csv', index_col=0)
    df = get_MA(ticker, df, 20)
    df = get_StdDev(ticker, df, 20, 2)

    return df
