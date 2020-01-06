# pyFinance
An attempt at using machine learning to determine when to by and sell S&amp;P 500 stocks

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
