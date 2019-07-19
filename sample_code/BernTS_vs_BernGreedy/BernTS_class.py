# -*- coding: utf-8 -*-

# Ken Sou 19/7/2019
# the function of this is same as BernTS but coded as 'class'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BernTS:
    def __init__(self,dataset):
        self.dataset  = dataset
        self.N_action, self.N_bandit = np.shape(dataset)
        self.ads_selected = []           # Action list, for statistical analysis.
        self.numbers_of_rewards_1 = [0] * self.N_bandit # the reward_1 of ad_i
        self.numbers_of_rewards_0 = [0] * self.N_bandit # the reward_0 of ad_i
        self.total_reward = 0  
    
    def run(self):
        for i in range(0,self.N_action):
            ad = 0
            zeta_list = np.zeros(self.N_bandit)

            zeta_list = self.sample_model(zeta_list)
            ad, reward = self.select_and_apply_action(ad, zeta_list)
            self.update_distribution(reward)
        print('the total reward of TS is {}'.format(total_reward))
        #end
        
    def sample_model(self,zeta_list):
        import random
        for i in range(0,self.N_bandit):
            zeta = random.betavariate(self.numbers_of_rewards_1[i] + 1,\
                                      self.numbers_of_rewards_0[i] + 1 )
            zeta_list[i] = zeta
        return zeta_list
    def select_and_apply_action(self,ad,zeta_list):
        ad = np.argmax(zeta_list)    # x_t <- argmax_k (zeta_k)
        self.ads_selected.append(ad)
        reward = dataset.values[n, ad]  # apply the action and observe reward_t.
        return ad, reward
    def update_distribution(self,reward):
        if reward == 1:
            self.numbers_of_rewards_1[ad] = self.numbers_of_rewards_1[ad] + 1
        else:
            self.numbers_of_rewards_0[ad] = self.numbers_of_rewards_0[ad] + 1
            self.total_reward = self.total_reward + reward

if __name__ == '__main__':
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
    # the shape of the dataset is needed to be (num_actions,num_bandit)
    TS = BernTS(dataset = dataset) 
    TS.run()
    