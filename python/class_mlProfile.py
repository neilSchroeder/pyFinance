import features as f #file containing the defined features
import pandas as pd

class mlProfile:
    """ produces a class which contains all the relevant information needed for
    performing an ml analysis of a given ticker.

    ticker: the ticker name for the profile to uses
    xDays: the number of days which will be passed to the function 'getFeature_XDayPctChange'
    """

    def __init__(self, ticker, xDays):
        self.ticker = ticker
        self.numDays = xDays
        self.df = pd.DataFrame()
        #get the features
        #basic x day percent change indicator
        df_temp = f.getFeature_XDayPctChange(ticker,xDays)
        use_cols =  df_temp.columns.difference(self.df)
        self.df = pd.merge(self.df, df_temp[use_cols], left_index=True, right_index=True, how='outer')

        #Moving average convergence-divergence
        #see README.md or features.py for details
        df_temp = f.getFeature_MACD(ticker)
        use_cols =  df_temp.columns.difference(self.df)
        self.df = pd.merge(self.df, df_temp[use_cols], left_index=True, right_index=True, how='outer')

        #McGinley Dynamic Indicator Convergence-Divergence
        #see README.md or features.py for more details
        df_temp = f.getFeature_MDICD(ticker)
        use_cols =  df_temp.columns.difference(self.df)
        self.df = pd.merge(self.df, df_temp[use_cols], left_index=True, right_index=True, how='outer')
        #McClellan Oscillator
        #see README.md or features.py for more details
        #note: no ticker is provided because this is calculated over the entrie S&P 500
        df_temp = f.getFeature_McClellanOscillator()
        use_cols =  df_temp.columns.difference(self.df)
        self.df = pd.merge(self.df, df_temp[use_cols], left_index=True, right_index=True, how='outer')
        #Bollinger Bands:
        #see README.md or features.py for more details
        #df_temp = f.getFeature_BollingerBands(ticker)
        #use_cols =  df_temp.columns.difference(self.df)
        #self.df = pd.merge(self.df, df_temp[use_cols], left_index=True, right_index=True, how='outer')
