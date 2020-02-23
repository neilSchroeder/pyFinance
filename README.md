# pyFinance
An attempt at using machine learning to determine when to by and sell S&amp;P 500 stocks.

##Strategy
By using a number of indicators, and avoiding coarse daily information,
 the algorithm will attempt to classify the data into whether the data will move above, 
 stay within, or fall below the current price by x% over the next 7 days. I'll be tuning 
 x%, possibly per stock, for performance of the network.

##Network
   Current I'm using a pretty simple voting classifier setup out of skLearn. The networks
   I'm considering, but don't always use, are 
   `[('lsvc', svm.LinearSVC()),
      ('nsvc', svm.NuSVC()),
      ('knn', neighbors.KNeighborsClassifier()),
      ('rfc', RandomForestClassifier(max_depth=5, n_estimators=50, random_state=1)),
      ('gnb', GaussianNB()),
      ('log', LogisticRegression(random_state=1)),
      ('gpc', GaussianProcessClassifier(1.0* RBF(1.0))),
      ('dtc', DecisionTreeClassifier(max_depth=5)),
      ('mlp', MLPClassifier(alpha=1, max_iter=1000)),
      ('ada', AdaBoostClassifier()),
      ('qda', QuadraticDiscriminantAnalysis())]`

# Indicators/Features
A list of current and future indicators and features to be used in the training of the dataset

1) 7 Day Percent: as name describes with a cut of +/- 3% for signaling
2) MACD: moving average convergence divergence. Defined to be the
12 Day EMA - 26 Day EMA. This feature signals when the current value crosses
  its 9 Day EMA.
3) McClellan Oscillator: global feature (calculated for all of S&P500) which measures
  the volatility of the adjusted net advances (advances - declines)/(advances+declines).
  This oscillator signals on when the 19 day EMA of ANAs crosses the 39 day EMA of ANAs
4) McGinley Dynamic Indicator Convergence-Divergence: this is an indicator of my own
  devising. It uses the McGinley Dynamic Indicator (MDI) and signals on the convergence
  divergence signal. Namely this calculates the MACD using the MDI instead of EMAs.
  Specifically I've implemented a 6 Day MDI - 13 day MDI. I've included some information
  on how the MACD and my MDICD compare.
5) Bollinger Bands
