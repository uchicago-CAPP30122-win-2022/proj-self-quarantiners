from selfq.data import apidata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import json

BASE_DIR = os.path.join(os.path.dirname( __file__ ), '..' )
DATA_DIR = os.path.join(BASE_DIR, "data")

def pre_covid():
    '''
    Read and preprocess the COVID datafile
    Return : 
        covid_df2 : (obj) A Pandas dataframe 
    '''
    covid_csv = os.path.join(DATA_DIR, 'owid-covid-data.csv')
    covid_df = pd.read_csv(covid_csv)
    covid_df['date'] = pd.to_datetime(covid_df['date'])
    date_val = covid_df['date'].unique()
    new_dict = {}
    new_dict['Date'] = []
    new_dict['New cases'] = []
    new_dict['New deaths'] = []
    new_dict['Total cases'] = []
    new_dict['Total deaths'] = []
    new_dict['Fully vaccinated'] = []
    
    for date in date_val:
        daily_new = 0
        daily_deaths = 0
        daily_total = 0
        daily_total_deaths = 0
        daily_vacc = 0
        daily_new = covid_df.loc[covid_df['date'] == date, 'new_cases'].sum()
        daily_deaths = covid_df.loc[covid_df['date'] == date, 'new_deaths'].sum()
        daily_total = covid_df.loc[covid_df['date'] == date, 'total_cases'].sum()
        daily_total_deaths = covid_df.loc[covid_df['date'] == date, 'total_deaths'].sum()
        daily_vacc = covid_df.loc[covid_df['date'] == date, 'people_fully_vaccinated'].sum()
        
        new_dict['Date'].append(date)
        new_dict['New cases'].append(daily_new)
        new_dict['New deaths'].append(daily_deaths)
        new_dict['Total cases'].append(daily_total)
        new_dict['Total deaths'].append(daily_total_deaths)
        new_dict['Fully vaccinated'].append(daily_vacc)
        
    covid_df2 = pd.DataFrame.from_dict(new_dict)
    return covid_df2

def plot_covid():
    '''
    Plot COVID data graphs
    Return : 
        fig : (obj) A Matplotlib figure 
    '''
    fig, axs = plt.subplots(5, 1, figsize=(15,10))
    fig.suptitle('World COVID-19 dashboard', size=15)
    axs[0].set_ylabel('Daily new cases', size=7)
    axs[0].set_title('Daily new cases (Worldwide)', size=10)
    axs[1].set_ylabel('Daily new deaths', size=7)
    axs[1].set_title('Daily new deaths (Worldwide)', size=10)
    axs[2].set_ylabel('Daily total cases', size=7)
    axs[2].set_title('Daily total cases (Worldwide)', size=10)
    axs[3].set_ylabel('Daily total deaths', size=7)
    axs[3].set_title('Daily total deaths (Worldwide)', size=10)
    axs[4].set_ylabel('Fully vaccinated people', size=7)
    axs[4].set_title('Fully vaccinated people (Worldwide)', size=10)
    
    axs[0].plot(covid_df2['Date'], covid_df2['New cases'], color='green')
    axs[1].plot(covid_df2['Date'], covid_df2['New deaths'], color='yellow')
    axs[2].plot(covid_df2['Date'], covid_df2['Total cases'], color='blue')
    axs[3].plot(covid_df2['Date'], covid_df2['Total deaths'], color='brown')
    axs[4].plot(covid_df2['Date'], covid_df2['Fully vaccinated'], color='red')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)
    return fig

def pre_safe():
    '''
    Read and preprocess the safe asset datafiles
    Return : 
        trs_df, gold_df, eur_df, jpy_df, gbp_df : (obj) Pandas dataframes
    '''
    treasury_csv = os.path.join(DATA_DIR, 'treasury.csv')
    trs_df = pd.read_csv(treasury_csv)
    trs_df['Date'] = pd.to_datetime(trs_df['Date'])
    trs_df = trs_df[~(trs_df['Date'] < '2020-01-01')]

    gold_csv = os.path.join(DATA_DIR, 'gold.csv')
    gold_df = pd.read_csv(gold_csv)
    gold_df['Date'] = pd.to_datetime(trs_df['Date'])
    gold_df = gold_df[~(gold_df['Date'] < '2020-01-01')]

    eur_csv = os.path.join(DATA_DIR, 'eurusd.csv')
    eur_df = pd.read_csv(eur_csv)
    eur_df['Date'] = pd.to_datetime(eur_df['Date'])
    eur_df = eur_df[~(eur_df['Date'] < '2020-01-01')]

    jpy_csv = os.path.join(DATA_DIR, 'jpyusd.csv')
    jpy_df = pd.read_csv(jpy_csv)
    jpy_df['Date'] = pd.to_datetime(jpy_df['Date'])
    jpy_df = jpy_df[~(jpy_df['Date'] < '2020-01-01')]

    gbp_csv = os.path.join(DATA_DIR, 'gbpusd.csv')
    gbp_df = pd.read_csv(gbp_csv)
    gbp_df['Date'] = pd.to_datetime(gbp_df['Date'])
    gbp_df = gbp_df[~(gbp_df['Date'] < '2020-01-01')]

    return trs_df, gold_df, eur_df, jpy_df, gbp_df

def plot_safe():
    '''
    Plot safe asset data graphs
    Return : 
        fig : (obj) A Matplotlib figure 
    '''
    fig, axs = plt.subplots(5, 1, figsize=(15,10))
    fig.suptitle('Price and yield of safe assets', size=15)

    axs[0].set_ylabel('Yield rate', size=7)
    axs[0].set_title('US Treasury(10YR) yield', size=10)

    axs[1].set_ylabel('Gold price', size=7)
    axs[1].set_title('Gold price (GC:CMX)', size=10)

    axs[2].set_ylabel('EUR/USD', size=7)
    axs[2].set_title('EUR/USD F/X', size=10)

    axs[3].set_ylabel('JPY/USD', size=7)
    axs[3].set_title('JPY/USD F/X', size=10)

    axs[4].set_ylabel('GBP/USD', size=7)
    axs[4].set_title('GBP/USD F/X', size=10)

    axs[0].plot(trs_df['Date'], trs_df['10 YR'], color='green')
    axs[1].plot(gold_df['Date'], gold_df['Close/Last'], color='yellow')
    axs[2].plot(eur_df['Date'], eur_df['Close/Last'], color='blue')
    axs[3].plot(jpy_df['Date'], jpy_df['Close/Last'], color='brown')
    axs[4].plot(gbp_df['Date'], gbp_df['Close/Last'], color='red')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

    return fig

def pre_equity():
    '''
    Read and preprocess the equity indices datafile
    Return : 
        nasdaq_df, n100_df, dow_df, sp500_df : (obj) Pandas dataframes
    '''
    nasdaq_csv = os.path.join(DATA_DIR, 'nasdaq.csv')
    nasdaq_df = pd.read_csv(nasdaq_csv)
    nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'])
    nasdaq_df = nasdaq_df[~(nasdaq_df['Date'] < '2020-01-01')]

    n100_csv = os.path.join(DATA_DIR, 'n100.csv')
    n100_df = pd.read_csv(n100_csv)
    n100_df['Date'] = pd.to_datetime(n100_df['Date'])
    n100_df = n100_df[~(n100_df['Date'] < '2020-01-01')]

    dow_csv = os.path.join(DATA_DIR, 'dow.csv')
    dow_df = pd.read_csv(dow_csv)
    dow_df['Date'] = pd.to_datetime(dow_df['Date'])
    dow_df = dow_df[~(dow_df['Date'] < '2020-01-01')]

    sp500_csv = os.path.join(DATA_DIR, 'sp500.csv')
    sp500_df = pd.read_csv(sp500_csv)
    sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    sp500_df = sp500_df[~(sp500_df['Date'] < '2020-01-01')]

    return nasdaq_df, n100_df, dow_df, sp500_df

def plot_equity():
    '''
    Plot equity data graphs
    Return : 
        fig : (obj) A Matplotlib figure 
    '''
    fig, axs = plt.subplots(4, 1, figsize=(15,10))
    fig.suptitle('Major equity indices', size=15)

    axs[0].set_ylabel('Index', size=7)
    axs[0].set_title('NASDAQ Composite', size=10)

    axs[1].set_ylabel('Index', size=7)
    axs[1].set_title('NASDAQ 100', size=10)

    axs[2].set_ylabel('Index', size=7)
    axs[2].set_title('Dow Jones Industrial Average', size=10)

    axs[3].set_ylabel('Index', size=7)
    axs[3].set_title('S&P 500', size=10)

    axs[0].plot(nasdaq_df['Date'], nasdaq_df['Close/Last'], color='green')
    axs[1].plot(n100_df['Date'], n100_df['Close/Last'], color='yellow')
    axs[2].plot(dow_df['Date'], dow_df['Close/Last'], color='blue')
    axs[3].plot(sp500_df['Date'], sp500_df['Close/Last'], color='brown')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

    return fig

def pre_energy():
    '''
    Read and preprocess the energy price datafiles
    Return : 
        wti_df, brent_df, gas_df : (obj) Pandas dataframes 
    '''
    wti_json = os.path.join(DATA_DIR, 'wti.json')
    with open(wti_json, 'r') as fp:
        wti_data = json.load(fp)

    brent_json = os.path.join(DATA_DIR, 'brent.json')
    with open(brent_json, 'r') as fp:
        brent_data = json.load(fp)

    gas_json = os.path.join(DATA_DIR, 'gas.json')
    with open(gas_json, 'r') as fp:
        gas_data = json.load(fp)

    wti_date = []
    wti_price = []
    for i in wti_data["series"]:
        for day in i["data"]:
            date, price = day
            wti_date.append(date)
            wti_price.append(price)
    wti_tups = list(zip(wti_date, wti_price))
    wti_df = pd.DataFrame(wti_tups, columns = ['Date', 'WTI price'])
    wti_df['Date'] = pd.to_datetime(wti_df['Date'])
    wti_df = wti_df[~(wti_df['Date'] < '2020-01-01')]

    brent_date = []
    brent_price = []
    for i in brent_data["series"]:
        for day in i["data"]:
            date, price = day
            brent_date.append(date)
            brent_price.append(price)
    brent_tups = list(zip(brent_date, brent_price))
    brent_df = pd.DataFrame(brent_tups, columns = ['Date', 'Brent price'])
    brent_df['Date'] = pd.to_datetime(brent_df['Date'])
    brent_df = brent_df[~(brent_df['Date'] < '2020-01-01')]

    gas_date = []
    gas_price = []
    for i in gas_data["series"]:
        for day in i["data"]:
            date, price = day
            gas_date.append(date)
            gas_price.append(price)
    gas_tups = list(zip(gas_date, gas_price))
    gas_df = pd.DataFrame(gas_tups, columns = ['Date', 'Gas price'])
    gas_df['Date'] = pd.to_datetime(gas_df['Date'])
    gas_df = gas_df[~(gas_df['Date'] < '2020-01-01')]

    return wti_df, brent_df, gas_df

def plot_energy():
    '''
    Plot energy price data graphs
    Return : 
        fig : (obj) A Matplotlib figure 
    '''
    fig, axs = plt.subplots(3, 1, figsize=(15,10))
    fig.suptitle('Major energy prices', size=15)

    axs[0].set_ylabel('WTI Price', size=7)
    axs[0].set_title('West Texas Intermediate (WTI)', size=10)

    axs[1].set_ylabel('Brent Price', size=7)
    axs[1].set_title('Brent Crude Oil', size=10)

    axs[2].set_ylabel('Natural Gas Price', size=7)
    axs[2].set_title('Natural Gas (Henry Hub Spot)', size=10)

    axs[0].plot(wti_df['Date'], wti_df['WTI price'], color='green')
    axs[1].plot(brent_df['Date'], brent_df['Brent price'], color='yellow')
    axs[2].plot(gas_df['Date'], gas_df['Gas price'], color='blue')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)
    return fig

def pre_commodity():
    '''
    Read and preprocess the commodities datafiles
    Return : 
        wheat_df, corn_df, soybean_df, coffee_df, sugar_df : (obj) Pandas dataframes
    '''
    wheat_csv = os.path.join(DATA_DIR, 'wheat.csv')
    wheat_df = pd.read_csv(wheat_csv)
    wheat_df['Date'] = pd.to_datetime(wheat_df['Date'])
    wheat_df = wheat_df[~(wheat_df['Date'] < '2020-01-01')]

    corn_csv = os.path.join(DATA_DIR, 'corn.csv')
    corn_df = pd.read_csv(corn_csv)
    corn_df['Date'] = pd.to_datetime(corn_df['Date'])
    corn_df = corn_df[~(corn_df['Date'] < '2020-01-01')]

    soybean_csv = os.path.join(DATA_DIR, 'soybean.csv')
    soybean_df = pd.read_csv(soybean_csv)
    soybean_df['Date'] = pd.to_datetime(soybean_df['Date'])
    soybean_df = soybean_df[~(soybean_df['Date'] < '2020-01-01')]

    coffee_csv = os.path.join(DATA_DIR, 'coffee.csv')
    coffee_df = pd.read_csv(coffee_csv)
    coffee_df['Date'] = pd.to_datetime(coffee_df['Date'])
    coffee_df = coffee_df[~(coffee_df['Date'] < '2020-01-01')]

    sugar_csv = os.path.join(DATA_DIR, 'sugar.csv')
    sugar_df = pd.read_csv(sugar_csv)
    sugar_df['Date'] = pd.to_datetime(sugar_df['Date'])
    sugar_df = sugar_df[~(sugar_df['Date'] < '2020-01-01')]

    return wheat_df, corn_df, soybean_df, coffee_df, sugar_df

def plot_commodity():
    '''
    Plot commodities prices data graphs
    Return : 
        fig : (obj) A Matplotlib figure
    '''
    fig, axs = plt.subplots(5, 1, figsize=(15,10))
    fig.suptitle('Major commodities prices', size=15)

    axs[0].set_ylabel('Price', size=7)
    axs[0].set_title('Wheat price', size=10)
    
    axs[1].set_ylabel('Price', size=7)
    axs[1].set_title('Corn price', size=10)

    axs[2].set_ylabel('Price', size=7)
    axs[2].set_title('Soybean price', size=10)

    axs[3].set_ylabel('Price', size=7)
    axs[3].set_title('Coffee price', size=10)

    axs[4].set_ylabel('Price', size=7)
    axs[4].set_title('Sugar price', size=10)

    axs[0].plot(wheat_df['Date'], wheat_df['Close/Last'], color='green')
    axs[1].plot(corn_df['Date'], corn_df['Close/Last'], color='yellow')
    axs[2].plot(soybean_df['Date'], soybean_df['Close/Last'], color='blue')
    axs[3].plot(coffee_df['Date'], coffee_df['Close/Last'], color='brown')
    axs[4].plot(sugar_df['Date'], sugar_df['Close/Last'], color='red')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

    return fig

def merge_df():
    '''
    Merge all dataframes into one
    Return : 
        df_fin : (obj) A Pandas dataframe 
    '''
    df_fin = pd.merge(covid_df2, trs_df[['Date', '10 YR']], on='Date', how='left')
    df_fin.rename(columns={'10 YR': 'TRS 10YR'}, inplace=True)
    df_fin = pd.merge(df_fin, gold_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Gold spot'}, inplace=True)
    df_fin = pd.merge(df_fin, eur_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'EUR/USD'}, inplace=True)
    df_fin = pd.merge(df_fin, jpy_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'JPY/USD'}, inplace=True)
    df_fin = pd.merge(df_fin, gbp_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'GBP/USD'}, inplace=True)
    df_fin = pd.merge(df_fin, nasdaq_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'NASDAQ Composite'}, inplace=True)
    df_fin = pd.merge(df_fin, n100_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'NASDAQ100'}, inplace=True)
    df_fin = pd.merge(df_fin, dow_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'DJIA'}, inplace=True)
    df_fin = pd.merge(df_fin, sp500_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'S&P500'}, inplace=True)
    df_fin = pd.merge(df_fin, wti_df[['Date', 'WTI price']], on='Date', how='left')
    df_fin.rename(columns={'WTI price': 'WTI'}, inplace=True)
    df_fin = pd.merge(df_fin, brent_df[['Date', 'Brent price']], on='Date', how='left')
    df_fin.rename(columns={'Brent price': 'Brent'}, inplace=True)
    df_fin = pd.merge(df_fin, gas_df[['Date', 'Gas price']], on='Date', how='left')
    df_fin.rename(columns={'Gas price': 'Natural gas'}, inplace=True)
    df_fin = pd.merge(df_fin, wheat_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Wheat'}, inplace=True)
    df_fin = pd.merge(df_fin, corn_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Corn'}, inplace=True)
    df_fin = pd.merge(df_fin, soybean_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Soybean'}, inplace=True)
    df_fin = pd.merge(df_fin, coffee_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Coffee'}, inplace=True)
    df_fin = pd.merge(df_fin, sugar_df[['Date', 'Close/Last']], on='Date', how='left')
    df_fin.rename(columns={'Close/Last': 'Sugar'}, inplace=True)

    df_fin.dropna(inplace = True)
    df_fin = df_fin.sort_values(by='Date')

    return df_fin

covid_df2 = pre_covid()
trs_df, gold_df, eur_df, jpy_df, gbp_df = pre_safe()
nasdaq_df, n100_df, dow_df, sp500_df = pre_equity()
wti_df, brent_df, gas_df = pre_energy()
wheat_df, corn_df, soybean_df, coffee_df, sugar_df = pre_commodity()
df_fin = merge_df()

def regr_key(key):
    '''
    Perform and plot linear regressions
    Input : 
        key : (str) A specific asset name
    Return : 
        fig : (obj) A Matplotlib figure 
    '''
    X = df_fin['New cases'].values.reshape(-1, 1)
    X2 = df_fin['New deaths'].values.reshape(-1, 1)
    y = df_fin[key].values.reshape(-1, 1)

    new_regr = LinearRegression().fit(X=X, y=y)
    new_regr2 = LinearRegression().fit(X=X2, y=y)

    fig, axs = plt.subplots(2, 2, figsize=(15,10))
    fig.suptitle('Regression on ' + key, size=15)

    axs[0,0].set_ylabel('Daily new cases', size=7, color='green')
    axs[0,0].set_title('Daily new cases vs. ' + key, size=10)

    axs[1,0].set_xlabel('Daily new cases', size=7, color='green')
    axs[1,0].set_ylabel(key, size=7, color='gold')
    axs[1,0].set_title('Regression : Daily new cases on ' + key, size=10)

    axs[0,1].set_ylabel('Daily new deaths', size=7, color='red')
    axs[0,1].set_title('Daily new deaths vs. ' + key, size=10)

    axs[1,1].set_xlabel('Daily new deaths', size=7, color='red')
    axs[1,1].set_ylabel(key, size=7, color='gold')
    axs[1,1].set_title('Regression : Daily new deaths on ' + key, size=10)

    axs[0,0].plot(df_fin['Date'], df_fin['New cases'], color='green')
    axs[0,0].tick_params(axis='x', labelsize=7)
    new_r2 = 'R^2 = ' + str(round(new_regr.score(X, y), 4))
    new_equation = 'Y = ' + str(np.round_(new_regr.coef_, 10)) + ' * X + ' + str(np.round_(new_regr.intercept_, 3))
    ax02 = axs[0,0].twinx()
    ax02.plot(df_fin['Date'], df_fin[key], color='gold')
    ax02.set_ylabel(key, size=7, color='gold')
    
    axs[1,0].scatter(X, y, color ='black')
    new_pred = new_regr.predict(X)
    axs[1,0].plot(X, new_pred, color="blue")
    if key == 'Natural gas':
        axs[1,0].text(0.65, 0.75, new_r2, ha='left', va='baseline', size=7, color='blue', transform=axs[1,0].transAxes)
        axs[1,0].text(0.65, 0.7, new_equation, ha='left', va='baseline', size=7, color='blue', transform=axs[1,0].transAxes)    
    else:
        axs[1,0].text(0.65, 0.15, new_r2, ha='left', va='baseline', size=7, color='blue', transform=axs[1,0].transAxes)
        axs[1,0].text(0.65, 0.10, new_equation, ha='left', va='baseline', size=7, color='blue', transform=axs[1,0].transAxes)

    axs[0,1].plot(df_fin['Date'], df_fin['New deaths'], color='red')
    axs[0,1].tick_params(axis='x', labelsize=7)
    death_r2 = 'R^2 = ' + str(round(new_regr2.score(X2, y), 4))
    dth_equation = 'Y = ' + str(np.round_(new_regr2.coef_, 8)) + ' * X + ' + str(np.round_(new_regr2.intercept_, 3))
    ax12 = axs[0,1].twinx()
    ax12.plot(df_fin['Date'], df_fin[key], color='gold')
    ax12.set_ylabel(key, size=7, color='gold')
    
    axs[1,1].scatter(X2, y, color ='black')
    new_pred2 = new_regr2.predict(X2)
    axs[1,1].plot(X2, new_pred2, color="blue")
    if key == 'JPY/USD' or key == 'Natural gas':
        axs[1,1].text(0.65, 0.75, death_r2, ha='left', va='baseline', size=7, color='blue', transform=axs[1,1].transAxes)
        axs[1,1].text(0.65, 0.7, dth_equation, ha='left', va='baseline', size=7, color='blue', transform=axs[1,1].transAxes)
    else:
        axs[1,1].text(0.65, 0.15, death_r2, ha='left', va='baseline', size=7, color='blue', transform=axs[1,1].transAxes)
        axs[1,1].text(0.65, 0.10, dth_equation, ha='left', va='baseline', size=7, color='blue', transform=axs[1,1].transAxes)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)

    return fig