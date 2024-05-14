from tvDatafeed import TvDatafeed, Interval



def stockData(symbol):

    tv = TvDatafeed()
    stock_data = tv.get_hist(symbol=symbol, exchange='BIST', interval=Interval.in_4_hour, n_bars=500000)['close']
    return stock_data
