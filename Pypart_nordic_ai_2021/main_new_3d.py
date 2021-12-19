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

price_change=pd.read_csv("price_change_m.csv")
# %%
hist_price = pd.read_csv("hist_price_2020_m.csv")

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
    'T': 696,
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
initial_state = {'price': exog_params_new['hist_price'][0],
                     'energy_amount':0 }
decision_variable = ['buy', 'hold', 'sell']
possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]

# %%
M = ESM(state_variable, decision_variable, initial_state, params_new, exog_params_new,possible_decisions)
P = ESP(M, policy_names)

# %%
discrete_energy = np.array([0.,1.])
        #discrete_energy = np.arange(0,1,0.2)
        #discrete_energy = np.array([0,0.5,1])
        # make list of prices with different increments
min_price = np.min(exog_params_new['hist_price'])
max_price = np.max(exog_params_new['hist_price'])

# %%
#inc_new = params_new['price_disc_list']
#inc = params['price_disc_list']
inc = 100
discrete_prices = np.arange(min_price,max_price+inc,inc)

print("\nStarting BackwardDP 3D")
# %%
state_variable_3 = ['price', 'energy_amount','prev_price']

index = bisect(discrete_prices, exog_params_new['hist_price'][1])
adjusted_p1 = discrete_prices[index]
index = bisect(discrete_prices, exog_params_new['hist_price'][0])
adjusted_p0 = discrete_prices[index]
initial_state_3 = {'price': adjusted_p1,'energy_amount':params_new['R0'], 'prev_price':adjusted_p0}
                
M3 = ESM(state_variable_3, decision_variable, initial_state_3, params_new, exog_params_new,possible_decisions)
P3 = ESP(M3, policy_names)

# %%
test_3D = BDP(discrete_prices, discrete_energy, exog_params_new['discrete_price_change_list'], 
              exog_params_new['discrete_price_change_cdf'], params_new['T'], copy(M3))

t0 = time.time()
value_dict = test_3D.bellman()
t1 = time.time()
time_elapsed = t1-t0
print("Time_elapsed_3D_model={:.2f} secs.".format(time_elapsed))

# %%
t0 = time.time()
value_dict = test_3D.bellman()
t1 = time.time()
time_elapsed = t1-t0
print("Time_elapsed_3D_model={:.2f} secs.".format(time_elapsed))


# %%
tS=time.time()
contribution = P3.run_policy(test_3D, "bellman_policy", params_new['T'])
print("Contribution using BackwardDP 3D is {:.2f}. Finished in {:.2f}s".format(contribution["contribution"],time.time()-tS))