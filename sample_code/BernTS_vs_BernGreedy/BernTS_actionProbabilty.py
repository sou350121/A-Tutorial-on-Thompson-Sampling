# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


import random
N = 10000                   # 1000 people, considering it as 'k' 
d = 10                      # 10 ads, the number of bandit


# BernTS
ads_selected = []           # Action list, for statistical analysis.
ads_count = [0] * d   # the Cumulative Distribution Function of action
ads_count_list = []
numbers_of_rewards_1 = [0] * d # the reward_1 of ad_i
numbers_of_rewards_0 = [0] * d # the reward_0 of ad_i
total_reward = 0            # total reward
ads_count_prop_list = []
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
    ads_count[ad] += 1
    ads_count_prop = np.array(ads_count)/sum(np.array(ads_count))
    ads_count_prop_list.append(ads_count_prop)
    reward = dataset.values[n, ad]  # apply the action and observe reward_t.
    # update distribution:
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
print('the total reward of TS is {}'.format(total_reward))
print('the action token by TS are {}'.format(ads_count))

def plot_action_prop(ads_count_prop_list,title ='',N=N,d=d):
    ads_count_prop_list = np.mat(ads_count_prop_list)
    x = np.arange(10,N)
    fig1 = plt.figure(num='fig111111', figsize=(10, 3), dpi=75, facecolor='#FFFFFF', edgecolor='#0000FF')
    plt.title(title)
    plt.xlabel(u'time period')
    plt.ylabel(u'action probabilty')
    handles_x = ['ad_'+str(x) for x in list(range(0,10))]
    for i in range(d):
        handles_x[i], = plt.plot(x,ads_count_prop_list[x[0]:,i])
    label_x = ['ad_'+str(x) for x in list(range(0,10))]
    plt.legend(handles=handles_x,labels=label_x,loc='best')
    plt.savefig(title+'_actionProb.jpg', format='jpg', transparent=True, dpi=300, pad_inches = 0)
    plt.show()
    plt.close()

plot_action_prop(ads_count_prop_list,'Thompson sampling')

ads_selected = []           # Action list, for statistical analysis.
ads_count = [0] * d   # the Cumulative Distribution Function of action
ads_count_list = []
numbers_of_rewards_1 = [0] * d # the reward_1 of ad_i
numbers_of_rewards_0 = [0] * d # the reward_0 of ad_i
total_reward = 0            # total reward
ads_count_prop_list = []
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
    ads_count_prop = np.array(ads_count)/sum(np.array(ads_count))
    ads_count_prop_list.append(ads_count_prop)
    reward = dataset.values[n, ad]  # apply the action and observe reward_t.
    # update distribution:
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

print('the total reward of Greedy is {}'.format(total_reward))
print('the action token by Greedy are {}'.format(ads_count))

#for showing chinese in the graph
plot_action_prop(ads_count_prop_list,title ='Greedy algo')



