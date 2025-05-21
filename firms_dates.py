"""
Porject by: 
    Einav Kohanim
    ID: 211492855
"""

#import
import pandas as df
import yfinance as yf #yahoo finance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#functions

#   -----  helper functions ------ #


def download_helper(ticker, startDate, endDate):
    """
    A small helper function to download data from youtube finance.

    Parameters
    ----------
    ticker : string
        firm's identification string.
    startDate : string
        start downloadfing from this date, inclusive.
    endDate : string
        stop downloading from this date, inclusive.
    
        
    
        
    Returns
    -------
    TYPE: pandas.core.frame.DataFrame

    """
    
    str = ":::Downloading: {_ticker} -- from: {_start}, to: {_end}".format(_ticker=ticker, _start=startDate, _end=endDate)
    print(str)
    return yf.download(ticker, start=startDate, end=endDate, interval="1d")
 


def percent_return(stockList):
    """
        calculating the dailyYield of a given stock, with an option to change the values either % or decimal. Default is decimal.
        the equation being: P(t)/p(t-1) -1, where t = day
        used in the 'download' function.
        
    Parameters
    ----------
    stockList : pandas.core.series.Series


    Returns
    -------
    listCalc : list

    """
    
    
    
    #seeding the 1st line as 'nan', since the 1st row of the db is required to be deleted.
    listCalc = [float('nan')]
    
    #start at index=1, stop at value of List's total items.
    for index in range(1, len(stockList)):
        #rd = (pt/p(t-1) -1)/percent (if percent-format is chosen)
        rd = ((stockList[index] / stockList[index-1]) -1)
        listCalc.append(rd)    
    
    return listCalc



def decimal_irx_return(stockList):
    """
       
    the necessary calculation, based on the doc, to get the r_rf numbers.
    used in the 'download' function
    
    Parameters
    ----------
    stockList : pandas.core.series.Series.

    Returns
    -------
    listCalc : list
        returns a list where every element is formatted with the equation: ³⁶⁵√(1 + val/100)-1
    """
    
    #seeding the 1st line as 'nan', since the 1st row of the db is required to be deleted.
    listCalc = [float('nan')]
    
    for val in stockList[1:]:
        rd = ((1 + (val/100))**(1/365))-1
        listCalc.append(rd)
    
    return listCalc



def get_alphabeta(xList, yList):
    """
    Acquires the alpha (intercept) and beta ('slope' coefficient) of the data's approxmiation to a linear equation, a 1st degree polynomial.

    Parameters
    ----------
    xList : pandas.core.series.Series
        the "x" values, its:  r_firm - r_rf.
    yList : pandas.core.series.Series
        the "y" values, its: r_market - r_rf.

    Returns
    -------
    dict
        compacts the alpha and beta into an easy to read dict object.

    """
    model = np.polyfit(xList, yList, 1)
    
    return {
        #y = bx + a
        'beta': model[0],
        'alpha' : model[1]
    }


def get_sharpe(firm_return, rf):
    """
    

    Parameters
    ----------
    firm_return : pandas.core.series.Series
        formatted series object.
    rf : pandas.core.series.Series
        formatted series object.

    Returns
    -------
    numpy.float64
        returns the sharpe value of the given firm.

    """
    
    
    avg_firm = np.mean(firm_return)
    avg_rf = np.mean(rf)
    return (avg_firm - avg_rf)/firm_return.std()
    
    
def get_treynor(firm_return, rf, beta):
    """
    

    Parameters
    ----------
    firm_return : pandas.core.series.Series
        formatted series object.
    rf : pandas.core.series.Series
        formatted series object.
    beta : numpy.float64
        the firm's beta-coefficient.

    Returns
    -------
    numpy.float64
        returns the firm's treynor value.

    """
    
    
    avg_firm = np.mean(firm_return)
    avg_rf = np.mean(rf)
    return (avg_firm - avg_rf)/beta


def get_returns(dates, firm_price):
    """
    

    Parameters
    ----------
    dates : pandas.core.series.Series
        formatted series object.
    firm_price : pandas.core.series.Series
        pre-formatted series object, of the downloaded stock prices.

    Returns
    -------
    numpy.float64
        returns the yearly returns of the irx
    """
    
    length = len(dates)-1
    r_total = (firm_price[length]/firm_price[0]) - 1
    
    #get difference from last day to first day
    N = (dates[length] - dates[0]).days
    return ((1 + r_total)**(365/N))-1


#   ----    plotter helper functions     ------#
def set_time_format(axis, dates):
    """
    

    Parameters
    ----------
    axis : matplotlib.axes._axes.Axes
        the axis object, that's attached to the given figure (frame).
        The axis-object is used to plot specific charts into its respective location.
    dates : pandas.core.series.Series
        formatted etc etc.

    Returns
    -------
    axis : matplotlib.axes._axes.Axes
        the formatted axis objectm, with date-labels padding added, etc.

    """
    
    #sets the gap between Dates to be 1/6 of the total time by dividing total days by 6~ 
    N = (dates[len(dates)-1] - dates[0]).days
    axis.xaxis.set_major_locator(plt.MultipleLocator(N/6))
    
    #set the format of the x-axis Dates to be y-m-d~
    axis.xaxis.set_major_formatter(mdates.DateFormatter(fmt="%Y-%m-%d"))
    
    #rotate the dates to be in a 45 degree angle~
    axis.tick_params(axis='x', labelrotation=45)
    
    return axis



def set_label_format(xlabel, ylabel, title_label, font_size, axis):
    """
    

    Parameters
    ----------
    xlabel : string
        the label attached to the x-axis.
    ylabel : string
        the label attached to the y-axis.
    title_label : string
        the label attached to the top of the chart.
    font_size : int
        size, of the font. Yes.
    axis : matplotlib.axes._axes.Axes
        the given axis object that needs its labels formattin'.

    Returns
    -------
    axis : TYPE
        formatted axis object, with the desired label names and font sizes..

    """
    axis.set_xlabel(xlabel, fontsize=font_size/1.4, color="black")
    axis.set_ylabel(ylabel, fontsize=font_size/1.6, color="black")
    axis.set_title(title_label, fontsize=font_size)
    return axis
    
#-----------------HELPER PLOT FUNCTION END--------------------#
#--- PLOT FUNCTIONS --- #

def plot_create(figure, axs, db, firm_title):
    """

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        the figure object.
    axs : matplotlib.axes._axes.Axes
        an array of axis-objects ("square shaped").
    db : pandas.core.frame.DataFrame
        the dataframe DB that holds all the data we need to properly plot the data.
    firm_title : string
        the name of the current firm.

    """
    
    print(":::::Starting to plot a new chart.")
    font_size = 30
    #1 stock prices
    plot_stockprices(db['firm'], db['date'], firm_title, font_size, axs[0,0])

    #2 stock returns
    plot_stockreturns(firm_db['r_firm'], firm_db['date'], firm_title, font_size, axs[0,1])

    #3 histogram
    plot_histogram(firm_db['r_firm'], firm_title, font_size, axs[1,0])

    #4 the linear regression
    plot_linearregression(X, Y, firm_title, font_size, axs[1,1])
    
    #last formatting of the figure, and saving to file, to where the project is located
    figure.tight_layout()
    
    figure.savefig(firm_title+"-chart.png")
    

    

#-------


def plot_stockprices(firm_prices, dates, firm_title, font_size, axis):
    """

    Parameters
    ----------
    firm_prices : pandas.core.series.Series
        the pre-formatted Series object, with the firm's downloaded stock prices.
    dates : pandas.core.series.Series
        a list of all the dates that Yahoo had data on.
    firm_title : string
        the firm's name.
    font_size : int
        font~size~!.
    axis : matplotlib.axes._axes.Axes
        Axis object.

    """
    
    
    #formats current axs with desired labels
    axis = set_label_format("Date", "Price ($)", "Stock prices: "+firm_title, font_size, axis)

    #formats current axs with correct dateformat and padding between each label, and also rotate the date labels.
    axis = set_time_format(axis, dates)       

    axis.plot(dates, firm_prices)


def plot_stockreturns(firm_returns, dates, firm_title, font_size, axis):
    """

    Parameters
    ----------
    firm_prices : pandas.core.series.Series
        the pre-formatted Series object, with the firm's downloaded stock prices.
    dates : pandas.core.series.Series
        a list of all the dates that Yahoo had data on.
    firm_title : string
        the firm's name.
    font_size : int
        font~size~!.
    axis : matplotlib.axes._axes.Axes
        Axis object.
        
    plot's the firm's stock returns.
    """
    
    #formats current axs with desired labels
    axis = set_label_format("Date", "Returns (%)", "Stock returns: "+firm_title, font_size, axis)
    
    #formats current axs with correct dateformat and padding between each label, and also rotate the date labels.
    axis = set_time_format(axis, dates)
    
    axis.plot(dates, firm_returns)
    
def plot_histogram(firm_returns, firm_title, font_size, axis):
    """

    Parameters
    ----------
    firm_prices : pandas.core.series.Series
        the pre-formatted Series object, with the firm's downloaded stock prices.
    dates : pandas.core.series.Series
        a list of all the dates that Yahoo had data on.
    firm_title : string
        the firm's name.
    font_size : int
        font~size~!.
    axis : matplotlib.axes._axes.Axes
        Axis object.
        
    plots the firm's histogram.
    """
    
    axis = set_label_format("Returns", "Frequency", "Histogram of stock return: "+firm_title, font_size, axis)
    
    axis.hist(firm_returns, bins=50)
    
    

"""
Linear regression method taken from: https://stackoverflow.com/a/6148315
"""
def plot_linearregression(xList, yList, firm_title, font_size, axis):
    """
    

    Parameters
    ----------
    xList : pandas.core.series.Series
        a Series where each element is: r_firm-r_rf.
    yList : pandas.core.series.Series
        a Series where each element is: r_market-r_rf.
    firm_title : string
        the firm's name.
    font_size : int
        ~the~size~of~the~font.
    axis : matplotlib.axes._axes.Axes

    """
        
    model = np.polyfit(xList, yList, 1)
    predict = np.poly1d(model)    
    
    axis = set_label_format("Adjusted market returns", "Adjusted returns", "Returns vs market returns: "+firm_title, font_size, axis)
    
    axis.plot(xList, yList, 'bo', xList, predict(xList), '--k')
    
    
    
 #####\\\\ helper functions#####

def download(ticker, startDate, endDate):
    """

    Parameters
    ----------
    ticker : string
        the firm's identification string.
    startDate : string
        the date to start downloading data
    endDate : string
        the date to stop downloading data, inclusive.

    Returns
    -------
    dict
        returns a dict where the format is:
            [date, firm, SPY, irx].

    Get a combination of stock data for each firm: firm, spy500, irx.
    """
    
    firmData = download_helper(ticker, startDate, endDate)
    SPY = download_helper("SPY", startDate, endDate)
    irx = download_helper("^irx", startDate, endDate)
    
    #currently Date is an index for whatever reason -- set it to be an acessible column!
    firmData.reset_index(inplace=True)  
    
    #return the firm-object as a dict, init_db will build the final db from this object.
    #loc[:, 'column-name'] returns every row, of a specific column -- i.e: we only get the column we want.
    
    #return order: Date, Company's stock, SPY, irx
        
    
    return {
        'date': firmData['Date'],
        'firm': firmData["Adj Close"],
        'market': SPY['Adj Close'],
        'rf': irx['Adj Close']
           }

###########        



def init_db(dbObj):
    """
    

    Parameters
    ----------
    dbObj : dict
        a pre-preppred dict object, of objects. 
        contents are: [list of dates, list of firm stock prices, list of the market prices, list of the rf vals].

    Returns
    -------
    db : pandas.core.frame.DataFrame
        an init DB object.
        
    Sets up the DB: 
    Takes a dict prepped by 'download', for a specific firm+S&P+irx,
    and preps a specific dataframe object to handle and save calculations to.

    """
    #init the database 
    print(":::::Initializing new DB.")
    dates = df.to_datetime(dbObj['date'], format="%Y-%m-%d").dt.date #convert to datetime object from string
    dbList = list(zip(dates, dbObj['firm'], dbObj['market'], dbObj['rf']))
    db = df.DataFrame(dbList, columns=['date', 'firm', 'market', "rf"])
    
    """
    Calculating the extra 3 columns for:
        1. daily stock return
        2. daily SPY return
        3. daily IRX return (in decimal instead of %)
    """
    
    #stock yield:
    db.insert(loc=len(db.columns), column="r_firm",     value= percent_return(db['firm']) )
    db.insert(loc=len(db.columns), column="r_market",   value= percent_return(db['market']) )
    db.insert(loc=len(db.columns), column="r_rf",       value= decimal_irx_return(db['rf']) )
    #drop all rows with NaN (first row guaranteed)
    db = db.dropna()
    db = db.reset_index(drop=True) #reset 1st index from 1 to 0, and so on
    
    return db
    
#END FUNC






#run program

#variables
config = df.read_csv("firms_dates.csv")
df_list_firm_names = []
df_list_dates_start = []
df_list_dates_end = []
df_list_beta = []
df_list_alpha = []
df_list_sharpe = []
df_list_treynor = []
df_list_returns = []

"""
Main code chunk: run a loop that downloads each firm's, and each SPY and IRX data separately (according to date),
process it separately, save results to a list, and then repeat the process for each firm until we get zip them all together.
FORMAT TO ZIP IS
    firmname | start | end | alpha | beta | sharpe | trenor | returns
"""

#iterates without adding the index into the row object
for row in config.itertuples(index=False):
    
    #CHART:
    #instantiate a new figure and axis objects for every row, a method to "flush out" the previous results, to prevent drawing over older plots
    figure, axs = plt.subplots(2, 2, figsize=(26,15))
    
   # get the date-range values of the firm, S&P500 & IRX in a single dict object
    firmObject = download(row.ticker, row.start, row.end)
   # generate the dataFrame DB from these propereties:
    firm_db = init_db(firmObject)
    
    df_list_firm_names.append(row.firm)
    df_list_dates_start.append(firm_db['date'][0])
    length = len(firm_db['date'])-1
    df_list_dates_end.append(firm_db['date'][length])
    
   #get alpha & beta
    Y = (firm_db['r_firm']-firm_db['r_rf'])
    X = (firm_db['r_market']-firm_db['r_rf'])
    (alpha, beta) = map(get_alphabeta(X, Y).get, ('alpha', 'beta')) #unpacking method with map
    df_list_alpha.append(alpha)
    df_list_beta.append(beta)
    
    #get sharpe
    df_list_sharpe.append(get_sharpe(firm_db['r_firm'], firm_db['r_rf']))
    
    #get treynor
    df_list_treynor.append(get_treynor(firm_db['r_firm'], firm_db['r_rf'], beta))
    
    #get yearly avg return of firm
    df_list_returns.append (get_returns(firm_db['date'], firm_db['firm']))
    
    #generate respective firm's chart!
    plot_create(figure, axs, firm_db, row.firm)
    
#---- main loop ends ---- #
"""
from here we can zip together all the data we collected into one big list to-save-to-file!
"""

db_all_firms = df.DataFrame(zip( 
    df_list_firm_names,
    df_list_dates_start,
    df_list_dates_end,
    df_list_beta,
    df_list_alpha,
    df_list_sharpe,
    df_list_treynor,
    df_list_returns), columns=["firm", "start", "end", "beta", "alpha", "sharpe", "treynor", "returns"])

db_all_firms.to_csv("results.csv")