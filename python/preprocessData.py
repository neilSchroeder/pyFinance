import numpy as np
import pandas as pd
import pickle

def processDataForLabels(ticker):
    hm_days = 7
    df = pd.read_csv('data/sp500_joinedClose.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buySellHold(*args):
    cols = [ c for c in args ]
    requirement = 0.2
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extractFeatureSet(ticker):
    tickers, df = processDataForLabels(ticker)

    df['{}_tarte'.format(ticker)] = list(map(buySellHold,
                                              df['{}_1d'.format(ticker, i)],
                                              df['{}_2d'.format(ticker, i)],
                                              df['{}_3d'.format(ticker, i)],
                                              df['{}_4d'.format(ticker, i)],
                                              df['{}_5d'.format(ticker, i)],
                                              df['{}_6d'.format(ticker, i)],
                                              df['{}_7d'.format(ticker, i))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = 
