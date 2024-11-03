import json
import time
import math
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

def time_calculate(a,spent):
# 시간, 분, 초로 변환
    hours = int(spent // 3600)
    minutes = int((spent % 3600) // 60)
    seconds = spent % 60
    print(f"{a} : {hours:02}:{minutes:02}:{seconds:06.2f}")

def State_CNN(table_name):
    torch.manual_seed(42)
    #데이터 추출  
    state_json = json.loads(plantsim.get_value(table_name))
    converted_list = []
    
    for item in state_json:
        numeric_values = [float(value) for value in item.values()]
        converted_list.append(numeric_values)
        
    tensor_data = torch.tensor(converted_list, dtype=torch.float32)
    state_tensor = tensor_data.unsqueeze(0).unsqueeze(0)
    height, width = tensor_data.shape[0], tensor_data.shape[1]
    
    model = CNN_Model(height, width)
    feature = model(state_tensor)
    feature = feature.detach().numpy()[0][0]
    return feature

def c_max_chart(c_max_list):
    plt.figure(figsize=(8, 6))
    plt.plot(c_max_list, label='C_max Values', color='b', linestyle='-')
    plt.title('C_max Value Trend')
    plt.xlabel('Index')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def moving_average_chart(data, window_size):
    data_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(data_avg, label='Smoothed Rewards', color='b', linestyle='-')
    plt.title('Smoothed Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.legend()
    plt.show()

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
        self.epsilon_min = 0.001  # 최소 탐험 확률
        self.epsilon_decay = 10 ** (math.log10(self.epsilon_min) / self.num_episode)  # 탐험 확률 감소율
        self.learning_rate = 0.001  
        self.gamma = 1 - self.learning_rate  # 0.99  # 할인 계수
        self.batch_size = 100
        self.target_update_frequency = 100  # 타켓 없데이트 빈도
        self.update_counter = 50  # 업데이트 횟수
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
min_c_max = 41530

path_1 ="C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240819.spp"

plantsim = Plantsim(version='23.2', license_type='Educational', visible = True)
plantsim.load_model(path_1)
plantsim.set_path_context('.models.model')
plantsim.set_event_controller()

reward_list = []
c_max_list = []
action_table = []
state_table = []
reward_table = []
loss_table = []

agent = DQNAgent(num_episode=10000, state_size=3, action_size=len(action_dic))

Start_time = time.time()
for i in range(agent.num_episode):
    if i < 5000:
        if agent.epsilon<0.5:
            agent.epsilon = 0.5
    elif i < 8000:
        epsilon_decay = 10 ** (math.log10(0.01) /1998)  # 탐험 확률 감소율
        agent.epsilon *= epsilon_decay
    elif  i > 9998:
        agent.epsilon = 0
        
    torch.manual_seed(42)
    action_history = []
    reward_history = []
    state_history = []

    plantsim.reset_simulation()

    ########################Init##################################
    WIP = plantsim.get_value("WIP")
    next_state = np.concatenate(([WIP], np.array(State_CNN("WS_json")).flatten(), np.array(State_CNN("Car_json")).flatten())).reshape(1, -1)
    state_history.append(next_state)
    print(f"Episode {i}")

    total_reward = 0
    done = plantsim.get_value("Terminate")
    
    while not done:
        state = next_state
        reward = 0
        action = agent.act(state)
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
        WIP = plantsim.get_value("WIP")
        next_state = np.concatenate(([WIP], np.array(State_CNN("WS_json")).flatten(), np.array(State_CNN("Car_json")).flatten())).reshape(1, -1)
        
        if not done:
            reward = plantsim.get_value("Reward")
        else:
            c_max = plantsim.get_value("Cmax")
            if c_max < min_c_max:
                reward = (100000 - c_max)*1000
            else:
                reward = -9999999
        
        total_reward += reward
        
        action_history.append(action)
        state_history.append(next_state)
        reward_history.append(reward)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        
        #if agent.epsilon > 0.2:
            #agent.learning_rate = 0.001
            
        ########################Finish##################################
        
    training_history = agent.get_training_history()
    loss_table.append(training_history)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
        
    print(agent.epsilon)
    print(c_max)

    reward_list.append(reward)
    c_max_list.append(c_max)
    state_table.append(state_history)
    action_table.append(action_history)
    reward_table.append(reward_history)
    
    if i % 1000 == 0 or i == agent.num_episode-1:
        c_max_chart(c_max_list)

End_time = time.time()
Spent_time = End_time - Start_time

time_calculate("전체시간",Spent_time)
data = {'Reward': reward_list, 'C_Max': c_max_list}

#df1 = pd.DataFrame(data)
#df2 = pd.DataFrame(state_table)
#df3 = pd.DataFrame(action_table)
#df4 = pd.DataFrame(reward_table)
#loss_table = pd.DataFrame(loss_table).T
