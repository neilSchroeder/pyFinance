import datetime as dt
import matplotlib.pyplot as pyplot
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import bs4 as bs
import pickle
import requests
import os


#style.use('ggplot')

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

    with open('sp500tickers.pickle','wb') as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers

#save_sp500_tickers()

def getYahooData(reload_sp500 = False):

    if reload_sp500:
        tickers = saveSp500Tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2019,11,24)

    for ticker in tickers:
        if not os.path.exists('data/{}.csv'.format(ticker)):
            print(ticker)
            if ticker.find(".") != -1:
                ticker = ticker.replace('.','-')
            df = web.get_data_yahoo(ticker, start, end)
            df.to_csv('data/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

getYahooData()
