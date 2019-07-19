# -*- coding: utf-8 -*-

# Ken Sou 18/7/2019
# a essential and simple model of BernTS
import numpy as np
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N = 10000                   # 1000 people, considering it as 'k' 
d = 10                      # 10 ads, the number of bandit

# BernTS
ads_selected = []           # Action list, for statistical analysis.
numbers_of_rewards_1 = [0] * d # the reward_1 of ad_i
numbers_of_rewards_0 = [0] * d # the reward_0 of ad_i
total_reward = 0            # total reward
for n in range(0, N):        # for each observation
    ad = 0
    zeta_list = np.zeros(10)
    #sample model:
    for i in range(0, d):   # for each ad
        zeta = random.betavariate(numbers_of_rewards_1[i] + 1, 
                                         numbers_of_rewards_0[i] + 1) # zeta_i
                                            # zeta_i;   '+1' is for initial condition.
        zeta_list[i] = zeta
    #select and apply action:
    ad = np.argmax(zeta_list)    # x_t <- argmax_k (zeta_k)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]  # apply the action and observe reward_t.
    # update distribution:
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
print('the total reward of TS is {}'.format(total_reward))
