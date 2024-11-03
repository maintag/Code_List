# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:14:00 2024

@author: leewa
"""
import time
import copy
import random
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

data = pd.read_csv('C:/Users/leewa/Desktop/공정변수.csv',encoding='euc-kr')
x = data[['Dorosperse Red KKL', 'Dorosperse Blue KKL', 'Dorosperse B/K KKL',
       'Dorosperse Dark Grey KKL', 'Dorosperse Brown K-3LR',
       'Dianix Yellow AM-2R', 'Dianix Red AM-SLR', 'Dianix Blue AM-2G',
       'Dianix Black AM-SLR', 'Dianix Grey AM-SLR', 'Dianix Yellow Brown AM-R',
       'Dorosperse Yellow KKL', 'Dorosperse Black KKL','Dorosperse Red KKL_prop', 'Dorosperse Blue KKL_prop', 'Dorosperse B/K KKL_prop',
       'Dorosperse Dark Grey KKL_prop', 'Dorosperse Brown K-3LR_prop',
       'Dianix Yellow AM-2R_prop', 'Dianix Red AM-SLR_prop', 'Dianix Blue AM-2G_prop',
       'Dianix Black AM-SLR_prop', 'Dianix Grey AM-SLR_prop', 'Dianix Yellow Brown AM-R_prop',
       'Dorosperse Yellow KKL_prop', 'Dorosperse Black KKL_prop','배합_Sunsolt RM-340S', '배합_빙초산', 'Lab 염색 시작온도', 'Lab 염색 상승속도 #1',
       'Lab 염색 상승온도 #1', 'Lab 염색 상승온도 #1 유지시간', 'Lab 염색 상승속도 #2',
       'Lab 염색 상승온도 #2', 'Lab 염색 상승온도 #2 유지시간', 'Lab 염색 상승속도 #3',
       'Lab 염색 상승온도 #3', 'Lab 염색 상승온도 #3 유지시간', 'Lab 염색 하강속도 #1',
       'Lab 염색 하강온도 #1', 'Lab 염색 하강온도 #1 유지시간', 'Lab 염색 하강속도 #2',
       'Lab 염색 하강온도 #2', 'Lab 염색 하강온도 #2 유지시간', 'Lab 염색 하강속도 #3',
       'Lab 염색 하강온도 #3', 'Lab 염색 하강온도 #3 유지시간', 'Lab 염색 종료속도', 'Lab 염색 종료온도',
       'Lab 염색 종료온도 유지시간']]
y = data[['잔욕염색 검사_K/S']]
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, test_size=0.2, random_state=42)
estimator = GradientBoostingRegressor(n_estimators = 500,max_depth=8, min_samples_split=7,random_state=0)
estimator.fit(x_train,y_train)
y_pred = estimator.predict(x)

class State:
    def __init__(self):
        self.no = 0
        self.상승속도1 = 0.5
        self.상승속도2 = 0.5
        self.상승속도3 = 0.5
        self.상승온도3 = 120
        self.상승온도유지시간3 = 10
    
    def __iter__(self):
        yield self.상승속도1
        yield self.상승속도2
        yield self.상승속도3
        yield self.상승온도3
        yield self.상승온도유지시간3
        
        
    def __repr__(self):
        return f"상승속도1: {self.상승속도1}, 상승속도2: {self.상승속도2}, 상승속도3: {self.상승속도3}, 상승온도3: {self.상승온도3}, 상승온도유지시간3: {self.상승온도유지시간3}"

    def get_ks(self):
        statex = {
            'Dorosperse Red KKL': [0], 
            'Dorosperse Blue KKL': [2],
            'Dorosperse B/K KKL': [0], 
            'Dorosperse Dark Grey KKL': [0],
            'Dorosperse Brown K-3LR': [0], 
            'Dianix Yellow AM-2R': [0],
            'Dianix Red AM-SLR': [0], 
            'Dianix Blue AM-2G': [0],
            'Dianix Black AM-SLR': [0],
            'Dianix Grey AM-SLR': [0], 
            'Dianix Yellow Brown AM-R': [0],
            'Dorosperse Yellow KKL': [4], 
            'Dorosperse Black KKL': [0],
            'Dorosperse Red KKL_prop': [0], 
            'Dorosperse Blue KKL_prop': [0.333],
            'Dorosperse B/K KKL_prop': [0], 
            'Dorosperse Dark Grey KKL_prop': [0],
            'Dorosperse Brown K-3LR_prop': [0], 
            'Dianix Yellow AM-2R_prop': [0],
            'Dianix Red AM-SLR_prop': [0], 
            'Dianix Blue AM-2G_prop': [0],
            'Dianix Black AM-SLR_prop': [0],
            'Dianix Grey AM-SLR_prop': [0],
            'Dianix Yellow Brown AM-R_prop': [0], 
            'Dorosperse Yellow KKL_prop': [0.667],
            'Dorosperse Black KKL_prop': [0],
            '배합_Sunsolt RM-340S': [0.3],
            '배합_빙초산': [0.2], 
            'Lab 염색 시작온도': [40], 
            'Lab 염색 상승속도 #1': [self.상승속도1],
            'Lab 염색 상승온도 #1': [64],
            'Lab 염색 상승온도 #1 유지시간': [0], 
            'Lab 염색 상승속도 #2': [self.상승속도2],
            'Lab 염색 상승온도 #2': [100],
            'Lab 염색 상승온도 #2 유지시간': [0],
            'Lab 염색 상승속도 #3': [self.상승속도3],
            'Lab 염색 상승온도 #3': [self.상승온도3],
            'Lab 염색 상승온도 #3 유지시간': [self.상승온도유지시간3],
            'Lab 염색 하강속도 #1': [1.97222],
            'Lab 염색 하강온도 #1': [64],
            'Lab 염색 하강온도 #1 유지시간': [0],
            'Lab 염색 하강속도 #2': [0],
            'Lab 염색 하강온도 #2': [64],
            'Lab 염색 하강온도 #2 유지시간': [0],
            'Lab 염색 하강속도 #3': [0],
            'Lab 염색 하강온도 #3': [64],
            'Lab 염색 하강온도 #3 유지시간': [0],
            'Lab 염색 종료속도': [0],
            'Lab 염색 종료온도': [64],
            'Lab 염색 종료온도 유지시간': [0]}
        statex = pd.DataFrame(statex)
        state_ks = estimator.predict(statex)
        statex = pd.DataFrame(statex)
        return estimator.predict(statex)[0]


state = State()

state1_values = random.sample(range(5, 16), 11)
state2_values = random.sample(range(5, 16), 11)
state3_values = random.sample(range(500, 1701), 1201)
state4_values = random.sample(range(120, 141), 21)
state5_values = random.sample(range(1, 9), 8)

state1_values = [round(value * 0.1, 1) for value in state1_values]
state2_values = [round(value * 0.1, 1) for value in state1_values]
state3_values = [round(value * 0.001, 3) for value in state3_values]
state4_values = [value for value in state4_values]
state5_values = [value* 10 for value in state5_values]

##############
class QLearning:
    def __init__(self, alpha=0.5, gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.99
        self.q_table = {}
        self.rewards_table = {}
        self.state1_values = [round(value * 0.1, 1) for value in random.sample(range(5, 16), 11)]
        self.state2_values = [round(value * 0.1, 1) for value in random.sample(range(5, 16), 11)]
        self.state3_values = [round(value * 0.001, 3) for value in random.sample(range(500, 1701), 1201)]
        self.state4_values = [value for value in random.sample(range(120, 141), 21)]
        self.state5_values = [value*10 for value in random.sample(range(1, 9), 8)]
        self.actions = None
        self.no = 0
        
    def get_q_value(self, state, num, action):
        num +=1
        state_str = str(state).replace(" ", "")
        state_no =  "Action : "+str(num)
        action_str = str(action).replace(" ", "")

        state_key = state_str
        no_key = num
        action_key = action_str

        if (state_key,no_key,action_key) not in self.q_table:
            self.q_table[(state_key,no_key,action_key)] = 0
            self.rewards_table[(state_key,no_key, action_key)] = 0

        return self.q_table[(state_key,no_key,action_key)]
    
    def choose_action(self, state):
        num = state.no
        if np.random.uniform() < self.epsilon:
            if state.no  == 0:
                return random.choice(self.state1_values)
            elif state.no  == 1:
                return random.choice(self.state2_values)
            elif state.no  == 2:
                return random.choice(self.state3_values)
            elif state.no  == 3:
                return random.choice(self.state4_values)
            else:
                return random.choice(self.state5_values)
        else:
            if state.no  == 0:
                self.actions = self.state1_values
            elif state.no  == 1:
                self.actions = self.state2_values
            elif state.no  == 2:
                self.actions = self.state3_values
            elif state.no  == 3:
                self.actions = self.state4_values
            else:
                self.actions = self.state5_values
                   
            q_values = [self.get_q_value(state,num,action) for action in self.actions]
            max_q_value = max(q_values)
            if q_values.count(max_q_value) > 1:
                max_q_value = max(self.get_q_value(state,num,a) for a in self.actions)
                best_action = [a for a in self.actions if self.get_q_value(state,num, a) == max_q_value][0]
                return best_action
            else:
                return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state,done):
        num = state.no+1
        state_str = str(state).replace(" ", "")
        state_no =  "Action : "+str(num)
        action_str = str(action).replace(" ", "")

        state_key = state_str
        no_key = num
        action_key = action_str
        
        if done == True:
            self.q_table[state_key,5,action_key] = reward
        else:
            if state.no  == 0:
                td_target = reward + self.gamma * max([self.get_q_value(next_state, 2, a) for a in self.state2_values])
            elif state.no  == 1:
                td_target = reward + self.gamma * max([self.get_q_value(next_state, 3, a) for a in self.state3_values])
            elif state.no  == 2:
                td_target = reward + self.gamma * max([self.get_q_value(next_state, 4, a) for a in self.state4_values])
            elif state.no  == 3:
                td_target = reward + self.gamma * max([self.get_q_value(next_state, 5, a) for a in self.state5_values])
                
            td_error = td_target - self.get_q_value(state, num, action)
            
            if (state_key,no_key,action_key) not in self.q_table:
                self.q_table[(state_key,no_key,action_key)] = 0
                self.rewards_table[(state_key,no_key, action_key)] = 0
                self.q_table[state_key,no_key,action_key] += td_error
            
        self.rewards_table[state_key,no_key,action_key] = reward
        
    def get_reward(self, KS, new_state_ks):
                return 1-new_state_ks
        #if round(new_state_ks,4) > round(KS,4):
         #   return -1
        #else:
         #   return (1-new_state_ks)
        
    def get_q_table_as_dataframe(self):
        df = pd.DataFrame(list(self.q_table.keys()), columns=['State', 'Action Num', 'Action'])
        df['Q_value'] = self.q_table.values()
        return df

        
q_learning  = QLearning()

class QValue:
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes
        self.best_state = None
        self.best_ks_value = float('inf')
        self.best_e_value = float('inf')
        self.best_iteration = None
        self.data = pd.DataFrame()

    def run(self, q_learning):
        state = State()
        KS = state.get_ks()
        KS2 = state.get_ks()
        KS3 = state.get_ks()
        KS4 = state.get_ks()
        KS5 = state.get_ks()
        for episode in range(self.n_episodes):
            done = False
            state = State()
            
            action = q_learning.choose_action(state)
            new_state = copy.copy(state)
            new_state.상승속도1 = action
            reward = q_learning.get_reward(KS, new_state.get_ks())
            #if KS > new_state.get_ks():
            KS = new_state.get_ks()
            #KS = new_state.get_ks()
            q_learning.learn(state, action, reward, new_state,done)
            state = new_state
            state.no +=1
            
            action = q_learning.choose_action(state)
            new_state = copy.copy(state)
            new_state.상승속도2 = action
            reward = q_learning.get_reward(KS, new_state.get_ks())
            #if KS > new_state.get_ks():
            KS = new_state.get_ks()
            #KS = new_state.get_ks()
            q_learning.learn(state, action, reward, new_state,done)
            state = new_state
            state.no +=1
            
            action = q_learning.choose_action(state)
            new_state = copy.copy(state)
            new_state.상승속도3 = action
            reward = q_learning.get_reward(KS, new_state.get_ks())
            #if KS > new_state.get_ks():
            KS = new_state.get_ks()
            #KS = new_state.get_ks()
            q_learning.learn(state, action, reward, new_state,done)
            state = new_state
            state.no +=1
            
            action = q_learning.choose_action(state)
            new_state = copy.copy(state)
            new_state.상승온도3 = action
            reward = q_learning.get_reward(KS, new_state.get_ks())
            #if KS > new_state.get_ks():
             #   KS = new_state.get_ks()
            KS = new_state.get_ks()
            q_learning.learn(state, action, reward, new_state,done)
            state = new_state
            state.no +=1
            
            action = q_learning.choose_action(state)
            new_state = copy.copy(state)
            new_state.상승온도유지시간3 = action
            reward = q_learning.get_reward(KS, new_state.get_ks())
            #if KS > new_state.get_ks():
            #    KS = new_state.get_ks()
            #    print(KS)
            KS = new_state.get_ks()
            done = True
            q_learning.learn(state, action, reward, new_state,done)
            
            print(episode)
            print(new_state)
            print(q_learning.epsilon)
            print(new_state.get_ks())
            
            if episode %16 == 0 or episode==0:
                episode_data = [{'Episode': episode+1, 
                                 'state': str(new_state), 
                                 'epsilon': q_learning.epsilon,
                                 'KS_Value': new_state.get_ks(),
                                 'Reward': reward}]
                self.data = pd.concat([self.data, pd.DataFrame(episode_data)], ignore_index=True)
                q_learning.epsilon *=0.98


            
        return self.data
    
start_time = time.time()
q_learning = QLearning(alpha=0.005, gamma=0.995)
q_value = QValue(n_episodes= 4001)
df = q_value.run(q_learning)
end_time = time.time()
print("Best state:", q_value.best_state)
print("Best KS:", q_value.best_ks_value)
#df.to_csv("C:/Users/leewa/Desktop/학습결과/table- 2000.csv", encoding='euc-kr')

run_time = end_time - start_time

a=df["KS_Value"].min()
print(a)
selected_rows = df[df['KS_Value'] < 0.118]
a =q_learning.get_q_table_as_dataframe()
a.to_csv("C:/Users/leewa/Desktop/학습결과/22Qtable -result-4000.csv", encoding='euc-kr')
dataframe = q_value.data
dataframe.to_csv("C:/Users/leewa/Desktop/학습결과/22러닝레이트episode4000-.csv", encoding='euc-kr')
print(run_time)