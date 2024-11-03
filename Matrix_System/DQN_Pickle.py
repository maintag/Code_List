import re
import json
import time
import math
import dill
import random
import itertools
import numpy as np
import pandas as pd

from collections import deque
from plantsim.plantsim import Plantsim 
from plantsim.table import Table

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 정의된 피클파일 로드(time_calculate, c_max_chart,moving_average_chart, simulation_module, check_makespan, State_CNN)
pickle_path = 'C:/Users/Whan Lee/Desktop/Matrix System/Code/Pickle/'
simulation_path = "C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240917.spp"

with open(pickle_path+'function_names.txt', 'r') as f:
    function_names = f.read().splitlines()
    for function_name in function_names:
        with open(pickle_path + function_name + '.pkl', 'rb') as f:
            globals()[function_name] = dill.load(f)
            
class CNN_Model(nn.Module):
    torch.manual_seed(42)
    def __init__(self, height, width):
        super(CNN_Model, self).__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1)  # 입력 채널 수를 1로 설정
        self.fc1 = nn.Linear(16 * height * width, 32)  # height와 width는 WS 데이터의 높이와 너비
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16 * self.height * self.width)  # height와 width는 WS 데이터의 높이와 너비
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNNetwork(nn.Module):
    torch.manual_seed(42)
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 57)
        self.fc2 = nn.Linear(57, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, num_episode, state_size, action_size):
        self.num_episode = num_episode
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # 탐험 확률 초기값
        self.epsilon_min = 0.01  # 최소 탐험 확률
        self.epsilon_decay = 10 ** (math.log10(self.epsilon_min) / self.num_episode)  # 탐험 확률 감소율
        self.learning_rate = 0.001  
        self.gamma = 0.99 # 1 - self.learning_rate  # 할인 계수
        self.batch_size = 100
        self.target_update_frequency = 100  # 타켓 없데이트 빈도
        self.update_counter = 50  # 업데이트 횟수
        self.device = torch.device("cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # 손실값, 상태, 행동 등을 저장하기 위한 리스트 추가
        self.loss_history = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []

    def remember(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def update_target_model(self):
        torch.manual_seed(42)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = np.array(transitions, dtype=object)

        state_batch = torch.FloatTensor(np.vstack(batch[:, 0])).to(self.device)

        ##print("Input size:", state_batch.size())
        ##print("Weight size (fc1):", self.model.fc1.weight.size())

        action_batch = torch.LongTensor(list(batch[:, 1])).to(self.device)
        reward_batch = torch.FloatTensor(list(batch[:, 2])).to(self.device)
        next_state_batch = torch.FloatTensor(np.vstack(batch[:, 3])).to(self.device)
        done_batch = torch.BoolTensor(list(batch[:, 4])).to(self.device)

        # 현재 상태에 대한 Q-value 예측
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        # 다음 상태에서의 최대 Q-value 예측 (타겟 네트워크 사용)
        torch.manual_seed(42)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[~done_batch] = self.target_model(next_state_batch[~done_batch]).max(1)[0].detach()

        # 기대 Q-value 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 손실 계산 및 역전파
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # 타겟 네트워크 없데이트(스텝 50번에 1번 업데이트)
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.update_target_model()

        # 손실값 및 히스토리 기록
        self.loss_history.append(loss.item())

    def get_training_history(self):
        return [self.loss_history]
    
action_dic = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
state_dic = {0: "WS", 1: "Car"}

time_step = 4500
plantsim = simulation_module(simulation_path, time_step)
c_max_list = []

# 단일 우선순위 규칙 기반의 Makespan 확인
single_rule_makespan = check_makespan(plantsim,action_dic)
single_rule_makespan_list = single_rule_makespan[0]
min_cmax = single_rule_makespan[1]
min_makespan = min_cmax
print(f'Single rule makespan_list : {single_rule_makespan_list}')
print(f'Rule Number : {single_rule_makespan_list.index(min_cmax)} / Min Makespan : {min_cmax}')

agent = DQNAgent(num_episode=50000, state_size=3, action_size=len(action_dic))
Start_time = time.time()
for i in range(agent.num_episode):
    if i < 9*(agent.num_episode/10):
        agent.epsilon *= (10 ** (math.log10(agent.epsilon_min)/(5*agent.num_episode)))
    else:
        agent.epsilon *= (10 ** (math.log10(agent.epsilon_min)/(agent.num_episode/5)))
    if i>49900:
        agent.epsilon = 0.0001
        
    plantsim.reset_simulation()
    WIP = plantsim.get_value("WIP")/75
    next_state = torch.tensor([WIP, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")])
    done = plantsim.get_value("Terminate")
    print(f"Episode {i}")
    
    total_reward = 0
    done = plantsim.get_value("Terminate")
    action_list = []
    while not done:
        state = next_state
        reward = 0
        action = agent.act(state)
        action_list.append(action)
        ws_action, car_action = action_dic[action]
        
        plantsim.set_value("Path_Rule", ws_action)
        plantsim.set_value("Nextwork_Rule", car_action)
        plantsim.set_value("Rule_select", False)
        plantsim.start_simulation()

        while not plantsim.get_value("Rule_select"):
            done = plantsim.get_value("Terminate")
            if done:
                break
            pass
        
        ########################Learning##################################

        if not done:
            reward = plantsim.get_value("Current_Progress")+ plantsim.get_value("Reward_Progress")
        else:
            c_max = plantsim.get_value("Cmax")
            if c_max <= min_cmax:
                reward = (50000 - c_max)*0.001
            else:
                reward = -9999999
        
        WIP = plantsim.get_value("WIP")/75
        next_state = torch.tensor([WIP, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")])
        done = plantsim.get_value("Terminate")
        
        total_reward += reward

        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
    if c_max < min_makespan:
        min_makespan = c_max
        min_action_list = action_list
        
    training_history = agent.get_training_history()
        
    print(agent.epsilon)
    print(c_max)

    c_max_list.append(c_max)
    
    if i % 1000 == 0 or i == agent.num_episode-1:
        c_max_chart(c_max_list)
        moving_average_chart(c_max_list,100)

End_time = time.time()
Spent_time = End_time - Start_time
time_calculate(Spent_time)

#data = {'Reward': reward_list, 'C_Max': c_max_list}
#df1 = pd.DataFrame(data)
#df2 = pd.DataFrame(state_table)
#df3 = pd.DataFrame(action_table)
#df4 = pd.DataFrame(reward_table)
#loss_table = pd.DataFrame(loss_table).T

