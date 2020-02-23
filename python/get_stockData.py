import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import bs4 as bs
import pickle
import requests
import os


style.use('ggplot')

#start = dt.datetime(2000,1,1)
#end = dt.datetime(2019,11,24)

#df = web.get_data_yahoo('TSLA',start,end)
#df.to_csv('data/tsla.csv')

def saveSp500Tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())

    with open('pickle/sp500tickers.pickle','wb') as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers

#save_sp500_tickers()

def getYahooData(reload_sp500 = False):

    if reload_sp500:
        tickers = saveSp500Tickers()
    else:
        with open("pickle/sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    start = dt.datetime(2009,1,1)
    end = dt.datetime.today()

    for ticker in tickers:
        if not os.path.exists('data/{}.csv'.format(ticker)):
            print(ticker)
            if ticker.find(".") != -1:
                ticker = ticker.replace('.','-')
            df = web.get_data_yahoo(ticker, start, end)
            df.to_csv('data/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

#getYahooData()

def compileData_joinedClose():
    with open("pickle/sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        if ticker.find("."):
            ticker = ticker.replace(".","-")
        df = pd.read_csv('data/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.rename(columns = {'Adj Close':ticker}, inplace = True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('data/sp500_joinedClose.csv')

#compileData()

def visualizeData():
    df = pd.read_csv('data/sp500_joinedClose.csv')
    df_corr = df.corr()

    data = df_corr.values

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor = False )
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor = False )
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

saveSp500Tickers()
getYahooData()
compileData_joinedClose()
visualizeData()
