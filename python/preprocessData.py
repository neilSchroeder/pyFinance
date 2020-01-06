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
    return tickers, df

def get_MA(ticker, days):
    weights = np.repeat(1., days)/days
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayMA'.format(ticker,days)] = z
    df.fillna(0,inplace=True)

    return tickers, df

def get_EMA(ticker, days):
    weights = np.exp(np.linspace(-1.,0., days))
    weights /= weights.sum()
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayEMA'.format(ticker,days)] = z

    return tickers, df

def get_EMAfromTicker(ticker, df, days):
    weights = np.exp(np.linspace(-1.,0., days))
    weights /= weights.sum()
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    z = np.convolve(df[ticker],weights)[:len(df[ticker])]
    z[:days] = z[days]
    df['{}_{}DayEMA'.format(ticker,days)] = z

    return tickers, df

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
    return tickers, df

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

    return tickers, df

def getSignal_XDayPctChange(*args):
    cols = [ c for c in args ]
    requirement = 0.03
    for col in cols:
        if col > requirement:
            return 1
        elif col < -requirement:
            return -1
    return 0

def getSignal_Crossover(a):
    ret = []
    last_val = 0
    for val in a:
        if val > 0 and last_val <= 0:
            ret.append(1)
        elif val < 0 and last_val >= 0:
            ret.append(-1)
        else:
            ret.append(0)
        last_val = val
    if len(ret) != len(a):
        print("lengths not equal")

    return ret

""" Exctracts the XDayPctChange indicator and prepares it for ml.
    Currently set up for a 1 week (7 Day) predictor """

def getFeature_XDayPctChange(ticker, days):
    tickers, df = get_XDayPctChange(ticker, days)

    df['{}_target0'.format(ticker)] = list(map(getSignal_XDayPctChange,
                                               *[df['{}_{}DayPctChange'.format(ticker, i)]for i in range(1, days+1)]))

    vals = df['{}_target0'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread {}DayPctChange:'.format(days), Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = df['{}_target0'.format(ticker)].values

    return x,y,df,tickers

""" Extracts the MACD indicator and prepares it for ml.
    MACD is defined to be the 12 Day EMA - 26 Day EMA
    The trigger for MACD is the point at which it crosses the 9 Day MACD EMA
"""

def getFeature_MACD(ticker):
    tickers12, df12 = get_EMA(ticker, 12)
    tickers, df = get_EMA(ticker, 26)

    df['{}_macd'.format(ticker)] = df12['{}_12DayEMA'.format(ticker)] - df['{}_26DayEMA'.format(ticker)]
    tickers_signalMacd, df_signalMacd = get_EMAfromTicker('{}_macd'.format(ticker), df, 9)

    df['{}_target1'.format(ticker)] =  df['{}_macd'.format(ticker)] - df_signalMacd['{}_macd_9DayEMA'.format(ticker)]

    vals = df['{}_target1'.format(ticker)].values.tolist()
    vals = getSignal_Crossover(vals)
    str_vals = [str(i) for i in vals]
    print('Data spread MACD:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]]
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = vals

    return x,y,df,tickers


def get_AdvancesAndDeclines():
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
        for ticker in tickers:
            if len(last_rows) > 1:
                if row[ticker] - last_rows[-2][ticker] > 0:
                    advances[-1] += 1
                elif row[ticker] - last_rows[-2][ticker] < 0:
                    declines[-1] += 1

    df['advances'] = np.array(advances)
    df['declines'] = np.array(declines)
    zm = np.array(advances)-np.array(declines)
    zp = np.array(advances)+np.array(declines)
    df['advMinusDec'] = np.array(advances) - np.array(declines)
    df['adjNetAdvances'] = np.divide(zm,zp)

    return tickers, df

""" Calculates the McClellan Oscillator or adjusted McClellan Oscillator
    for the S&P500. This oscillator is uses EMAs of the Adjusted Net Advances.
    The oscillator is defined to be the difference of the 19 Day EMA of ANA and
    the 39 Day EMA of ANA
"""

def getFeature_McClellanOscillator():
    """get the advances and declines"""
    tickers, df = get_AdvancesAndDeclines()
    tickers, df = get_EMAfromTicker('adjNetAdvances', df, 19)
    tickers, df = get_EMAfromTicker('adjNetAdvances', df, 39)
    df['McClellan'] = df['adjNetAdvances_19DayEMA'] - df['adjNetAdvances_39DayEMA']

    y = np.array(getSignal_Crossover(df['McClellan']))
    str_vals = [str(i) for i in y]
    print('Data Spread McClellan:', Counter(str_vals))
    x = df[[ticker for ticker in tickers]].replace([np.inf,-np.inf], 0)
    x.fillna(0,inplace=True)
    x = x.values

    return x, y, df, tickers

""" Calculates the MACD using the McGinley Dynamic Indicator.
    The rule of thumb seems to be use half the period of an EMA to get similar
    performance. So, there is a 6 day MDI and a 13 day MDI. Signaling is done on
    the difference.
"""

def getFeature_MDICD(ticker):
    tickers, df6 = get_MDI(ticker, 6)
    tickers13, df = get_MDI(ticker, 13)
    df['{}_mdicd'.format(ticker)] = df6['{}_6DayMDI'.format(ticker)] - df['{}_13DayMDI'.format(ticker)]

    vals = df['{}_mdicd'.format(ticker)].values.tolist()
    vals = getSignal_Crossover(vals)
    str_vals = [str(i) for i in vals]
    print('Data spread MDI:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]]
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    x = df_vals.values
    y = vals

    return x,y,df,tickers



def machineLearn(ticker):
    """get features from data"""
    x_7dpc, y_7dpc, df_7dpc, tickers_7dpc = getFeature_XDayPctChange(ticker,7)
    x_macd, y_macd, df_macd, tickers_macd = getFeature_MACD(ticker)
    x_mdicd, y_mdicd, df_mdicd, tickers_mdicd = getFeature_MDICD(ticker)
    x_mcClellan, y_mcClellan, df_mcClellan, tickers = getFeature_McClellanOscillator()


    """combine and prepare for splitting"""
    df = df_7dpc
    for col in df_macd:
        try:
            df[col]
        except:
            df[col] = df_macd[col]

    for col in df_mcClellan:
        try:
            df[col]
        except:
            df[col] = df_mcClellan[col]

    df_vals = df[[ticker for ticker in tickers_7dpc]]
    x = df_vals.values
    y_7dpc = df['{}_target0'.format(ticker)].values.tolist()
    if len(y_7dpc) != len(y_macd):
        for i in range(len(y_macd)-len(y_7dpc)):
            y_macd.pop(0)


    y = np.array(y_7dpc) + np.array(y_macd) + np.array(y_mcClellan)
    for i,tar in enumerate(y):
        if tar > 1:
            y[i] = 1
        elif tar < -1:
            y[i] = -1
        else:
            y[i] = 0

    str_vals = [str(i) for i in y]
    print('Data spread y:', Counter(str_vals))

    """Split the data"""
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,
                y,
                test_size = 0.2)

    """Declare our voting classifier and which classifiers it will use"""
    clf = VotingClassifier(
            estimators=[
                        #('lsvc', svm.LinearSVC()),
                        #('nsvc', svm.NuSVC()),
                        ('knn', neighbors.KNeighborsClassifier()),
                        #('rfc', RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)),
                        ('gnb', GaussianNB()),
                        #('log', LogisticRegression(random_state=1)),
                        #('gpc', GaussianProcessClassifier(1.0* RBF(1.0))),
                        #('dtc', DecisionTreeClassifier(max_depth=5)),
                        #('mlp', MLPClassifier(alpha=1, max_iter=1000)),
                        ('ada', AdaBoostClassifier()),
                        #('qda', QuadraticDiscriminantAnalysis())
                        ],
            voting='hard')

    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(x_test)

    print('{} Predicted spread:'.format(ticker), Counter(predictions))

    return confidence

getFeature_MDICD('XOM')
# machineLearn('MMM')
# for ticker in get_tickers():
#     machineLearn(ticker)
