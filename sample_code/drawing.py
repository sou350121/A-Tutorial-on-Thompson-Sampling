# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

def action_count(ads_selected):
    i = 0 
    count_list = []
    for ad in ads_selected:
        ads_count[i,ad] = 1
        ads_count = ads_count + ads_count
        i += 1
        
import random
N = 10000                   # 1000 people, considering it as 'k' 
d = 10                      # 10 ads, the number of bandit
ads_selected = []           # Action list, for statistical analysis.
ads_count = [0] * d   # the Cumulative Distribution Function of action
ads_count_list = []
numbers_of_rewards_1 = [0] * d # the reward_1 of ad_i
numbers_of_rewards_0 = [0] * d # the reward_0 of ad_i
total_reward = 0            # total reward

# BernGreedy 
for n in range(0, N):        # for each observation
    ad = 0
    zeta_list = np.zeros(10)
    #sample model:
    for i in range(0, d):   # for each ad
        zeta = (numbers_of_rewards_1[i] + 1)/(numbers_of_rewards_1[i]+ numbers_of_rewards_0[i] + 1) # zeta_i
        zeta_list[i] = zeta
    #select and apply action:
    ad = np.argmax(zeta_list)    # x_t <- argmax_k (zeta_k)
    ads_selected.append(ad)
    ads_count[ad] += 1
    ads_count_list.append(ads_count)
    reward = dataset.values[n, ad]  # apply the action and observe reward_t.
    # update distribution:
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

print('the total reward of Greedy is {}'.format(total_reward))
print('the action token by Greedy are {}'.format(ads_count))


# BernTS
ads_selected = []           # Action list, for statistical analysis.
ads_count = [0] * d   # the Cumulative Distribution Function of action
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
        zeta_list[i] = zeta
    #select and apply action:
    ad = np.argmax(zeta_list)    # x_t <- argmax_k (zeta_k)
    ads_selected.append(ad)
    ads_count[ad] += 1
    reward = dataset.values[n, ad]  # apply the action and observe reward_t.
    # update distribution:
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
print('the total reward of TS is {}'.format(total_reward))
print('the action token by TS are {}'.format(ads_count))



