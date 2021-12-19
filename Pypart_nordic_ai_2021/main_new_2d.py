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
    'nPriceChangeInc': 81,
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
inc = 10
discrete_prices = np.arange(min_price,max_price+inc,inc)

# %%
print("\nStarting BackwardDP 2D")
test_2D = BDP(discrete_prices, discrete_energy, exog_params_new['discrete_price_change_list'], 
              exog_params_new['discrete_price_change_cdf'], 
              params_new['T'], copy(M))
# %%
t0 = time.time()
value_dict = test_2D.bellman()
t1 = time.time()
time_elapsed = t1-t0

print("Time_elapsed_2D_model={:.2f} secs.".format(time_elapsed))
# %%
print("Starting policy evaluation for the actual sample path")
tS=time.time()
contribution = P.run_policy(test_2D, "bellman_policy", params_new['T'])
#%%
print("Contribution using BackwardDP 3D is {:.2f}. Finished in {:.2f}s".format(contribution["contribution"],time.time()-tS))
