
# packages
# %%
import os as os
import time
from collections import namedtuple, defaultdict
import numpy as np
import pandas as pd

# %%
from EnergyStorageModel import EnergyStorageModel as ESM
from EnergyStoragePolicy import EnergyStoragePolicy as ESP
from data_process import process_raw_price_data
from BackwardDP import BDP


# %%
from BackwardDP import BDP
# %%
import matplotlib.pyplot as plt
from copy import copy
from scipy.ndimage.interpolation import shift
import pickle
from bisect import bisect
import seaborn as sns

# %%
def process_raw_price_data(file,params):
    
    import time as time
    DISC_TYPE = "FROM_CUM"
    #DISC_TYPE = "OTHER"

    print("Processing raw price data. Constructing price change list and cdf using {}".format(DISC_TYPE))
    tS = time.time()

    # load energy price data from the Excel spreadsheet
    raw_data = pd.read_excel(file, sheet_name="Raw Data")

    # look at data spanning a week
    data_selection = raw_data.iloc[0:params['T'], 0:5]

    # rename columns to remove spaces (otherwise we can't access them)
    cols = data_selection.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    data_selection.columns = cols

    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')
    #print(sort_by_price.head())

    hist_price = np.array(data_selection['PJM_RT_LMP'].tolist())
    #print(hist_price[0])
    hist_price = hist_price[0:params['T']]

    max_price = pd.DataFrame.max(sort_by_price['PJM_RT_LMP'])
    min_price = pd.DataFrame.min(sort_by_price['PJM_RT_LMP'])
    print("Min price {:.2f} and Max price {:.2f}".format(min_price,max_price))
    



    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')

    # calculate change in price and sort values of change in price in ascending order
    data_selection['Price_Shift'] = data_selection.PJM_RT_LMP.shift(1)
    data_selection['Price_Change'] = data_selection['PJM_RT_LMP'] - data_selection['Price_Shift']
    sort_price_change = data_selection.sort_values('Price_Change')
    

    # discretize change in price and obtain f(p) for each price change
    max_price_change = pd.DataFrame.max(sort_price_change['Price_Change'])
    min_price_change = pd.DataFrame.min(sort_price_change['Price_Change'])
    print("Min price change {:.2f} and Max price change {:.2f}".format(min_price_change,max_price_change))
    
    
    

    # there are 191 values for price change
    price_changes_sorted = sort_price_change['Price_Change'].tolist()
    # remove the last NaN value
    price_changes_sorted.pop()

    if DISC_TYPE == "FROM_CUM":
    # discretize price change  by interpolating from cumulative distribution
        xp = price_changes_sorted
        fp = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(fp, 1)

        # obtain 30 discrete prices
        discrete_price_change_cdf = np.linspace(0, 1, int(params['nPriceChangeInc']))
        discrete_price_change_list = []
        for i in discrete_price_change_cdf:
            interpolated_point = np.interp(i, cum_fn, xp)
            discrete_price_change_list.append(interpolated_point)
    else:
        price_change_range = max_price_change - min_price_change
        price_change_increment = price_change_range / int(params['nPriceChangeInc'])
        discrete_price_change = np.arange(min_price_change, max_price_change, price_change_increment)
        discrete_price_change_list = list(np.append(discrete_price_change, max_price_change))


        f_p = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
        cum_fn = np.append(f_p, 1)
        discrete_price_change_cdf = []
        for c in discrete_price_change_list:
            interpolated_point = np.interp(c, price_changes_sorted, cum_fn)
            discrete_price_change_cdf.append(interpolated_point)

    price_changes_sorted = np.array(price_changes_sorted)
    discrete_price_change_list = np.array(discrete_price_change_list)
    discrete_price_change_cdf = np.array(discrete_price_change_cdf)
    discrete_price_change_pdf = discrete_price_change_cdf - shift(discrete_price_change_cdf,1,cval=0)

    mean_price_change = np.dot(discrete_price_change_list,discrete_price_change_pdf)

    #print("discrete_price_change_list ",discrete_price_change_list)
    #print("discrete_price_change_cdf",discrete_price_change_cdf)
    #print("discrete_price_change_pdf",discrete_price_change_pdf)


    print("Finishing processing raw price data in {:.2f} secs. Expected price change is {:.2f}. Hist_price len is {}".format(time.time()-tS,mean_price_change,len(hist_price)))
    #input("enter any key to continue...") 

    exog_params = {'hist_price':hist_price,"price_changes_sorted":price_changes_sorted,"discrete_price_change_list":discrete_price_change_list,"discrete_price_change_cdf":discrete_price_change_cdf}


    return exog_params

# %%
file = '/home/peymakorwork/Documents/PhD_UiS_2021/NordicAIMeet2021/Pypart_nordic_ai_2021/data/Parameters.xlsx'
seed = 189654913
# %%

#Reading the algorithm pars
# %%
parDf = pd.read_excel(file, sheet_name = 'ParamsModel')
parDict=parDf.set_index('Index').T.to_dict('list')
params = {key:v for key, value in parDict.items() for v in value} 
params['seed'] = seed
params['T'] = min(params['T'],192)
    

parDf = pd.read_excel(file, sheet_name = 'GridSearch')
parDict=parDf.set_index('Index').T.to_dict('list')
paramsPolicy = {key:v for key, value in parDict.items() for v in value}
params.update(paramsPolicy)

parDf = pd.read_excel(file, sheet_name = 'BackwardDP')
parDict=parDf.set_index('Index').T.to_dict('list')
paramsPolicy = {key:v for key, value in parDict.items() for v in value}
params.update(paramsPolicy)

if isinstance(params['priceDiscSet'], str):
    price_disc_list = params['priceDiscSet'].split(",")
    price_disc_list = [float(e) for e in price_disc_list]
    
else:
    
    price_disc_list = [float(params['priceDiscSet'])] 
params['price_disc_list']=price_disc_list

print("Parameters ",params)
    #input("enter any key to continue...")
    
# %%
exog_params = process_raw_price_data(file,params)


# %%
exog_params["hist_price"]

# %%
price_change=pd.read_csv("price_change.csv")
# %%
hist_price = pd.read_csv("hist_price_2020.csv")

# %%
exog_params_new = {
    "hist_price": np.array(hist_price["price"]),
    "discrete_price_change_list": np.array(price_change["price_change_quantile"]),
    "discrete_price_change_cdf": np.array(price_change["quantiles"])
}

## parameter values

# %%
params_new = {
    'Algorithm': 'BackwardDP',
    'T': 168,
    'eta': 1,
    'Rmax': 1,
    'R0': 1,
    'seed': 189654913,
    'nPriceChangeInc': 100,
    'priceDiscSet': 2,
    'run3D': 1,
    'price_disc_list': [500]
}

## policy information
# %%
policy_names = ['buy_low_sell_high_policy','bellman_policy']
state_variable = ['price', 'energy_amount']
initial_state = {'price': exog_params['hist_price'][0],
                     'energy_amount':0 }
decision_variable = ['buy', 'hold', 'sell']
possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]

# %%
M = ESM(state_variable, decision_variable, initial_state, params, exog_params,possible_decisions)
P = ESP(M, policy_names)

# %%
discrete_energy = np.array([0.,1.])
        #discrete_energy = np.arange(0,1,0.2)
        #discrete_energy = np.array([0,0.5,1])
        # make list of prices with different increments
min_price = np.min(exog_params['hist_price'])
max_price = np.max(exog_params['hist_price'])

# %%
#inc_new = params_new['price_disc_list']
#inc = params['price_disc_list']
inc = 0.5
discrete_prices = np.arange(min_price,max_price+inc,inc)

# %%
print("\nStarting BackwardDP 2D")

# %%
test_2D = BDP(discrete_prices, discrete_energy, exog_params['discrete_price_change_list'], 
              exog_params['discrete_price_change_cdf'], 
              params['T'], copy(M))
# %%
t0 = time.time()
value_dict = test_2D.bellman()
t1 = time.time()
time_elapsed = t1-t0

print("Time_elapsed_2D_model={:.2f} secs.".format(time_elapsed))
# %%
print("Starting policy evaluation for the actual sample path")
tS=time.time()
contribution = P.run_policy(test_2D, "bellman_policy", params['T'])
print("Contribution using BackwardDP 3D is {:.2f}. Finished in {:.2f}s".format(contribution["contribution"],time.time()-tS))
# %%

# Thre D case

print("\nStarting BackwardDP 3D")

# %%
state_variable_3 = ['price', 'energy_amount','prev_price']                
index = bisect(discrete_prices, exog_params['hist_price'][1])
adjusted_p1 = discrete_prices[index]
index = bisect(discrete_prices, exog_params['hist_price'][0])
adjusted_p0 = discrete_prices[index]
initial_state_3 = {'price': adjusted_p1,'energy_amount':params['R0'], 'prev_price':adjusted_p0}
# %%
M3 = ESM(state_variable_3, decision_variable, initial_state_3, params, exog_params,possible_decisions)
P3 = ESP(M3, policy_names)
# %%
test_3D = BDP(discrete_prices, discrete_energy, 
              exog_params['discrete_price_change_list'], exog_params['discrete_price_change_cdf'],
              params['T'], copy(M3))
# %%
t0 = time.time()
value_dict = test_3D.bellman()
t1 = time.time()
time_elapsed = t1-t0
print("Time_elapsed_3D_model={:.2f} secs.".format(time_elapsed))