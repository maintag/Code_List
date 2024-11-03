import itertools
import pandas as pd
import numpy as np
import torch.optim as optim
import random
import json
import time

from collections import deque
from plantsim.plantsim import Plantsim 
from plantsim.table import Table

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def time_calculate(a,spent):
# 시간, 분, 초로 변환
    hours = int(spent // 3600)
    minutes = int((spent % 3600) // 60)
    seconds = spent % 60
    print(f"{a} : {hours:02}:{minutes:02}:{seconds:06.2f}")

class CNN_Model(nn.Module):
    def __init__(self,height, width):
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


class State:
    def __init__(self, state_num, WIP, ws_state, car_state):
        self.state_num = state_num
        self.WIP = WIP
        self.ws_state = ws_state
        self.car_state = car_state


class Q_learning:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # 학습률
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # 탐험 확률
        self.q_table = {}  # Q-값 저장

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # 탐험
        else:
            return np.argmax(self.q_table[state])  # 최적 행동 선택
        
    def get_q_value(self, state, action):
        return self.q_table.get((state.state_num, state.WIP, state.ws_state,state.car_state, action), 0)

    def learn(self, state, action, action_count, reward, next_state, next_action):
        if next_state.state_num == action_count:
            final_state = State(next_state.state_num,next_state.WIP,next_state.ws_state,next_state.car_state)
            max_q_value = self.get_q_value(final_state, 9)
        else: 
            max_q_value = self.get_q_value(next_state, next_action)
        
        q_key = ((state.state_num,state.WIP,state.ws_state,state.car_state), action)
        
        if q_key not in self.q_table:
            self.q_table[q_key] = 0.0
        self.q_table[q_key] += self.alpha * (reward + self.gamma * max_q_value - self.q_table[q_key])

action_dic = {0: (0, 0), 1: (0, 1), 2: (0, 2)}
action_size = len(action_dic)
state_size = 5

columns = range(1, state_size+1)
values = range(action_size)

all_cases = list(itertools.product(values, repeat=len(columns)))
all_cases

path_1 ="C:/Users/leewa/Desktop/GM_Matrix_model_240723.spp"
#path_1 = "D:/Python/01_Project/GM_Matrix/GM_Matrix_model_240723.spp"

plantsim = Plantsim(version='23.2', license_type='Educational', visible = True)
plantsim.load_model(path_1)
plantsim.set_path_context('.models.model')
plantsim.set_event_controller()


agent = Q_learning(state_size,action_size,alpha=0.1, gamma=0.99, epsilon=0.1)

c_max_list= []
all_state = []
start_time = time.time()

for i in range(len(all_cases)):
    
    #데이터 저장
    state_data = []  
    action = all_cases[i]
    action_count = len(action)
    #시뮬레이션 초기화
    plantsim.set_value("Reset_check", False)
    plantsim.reset_simulation()
    
    while not plantsim.get_value("Reset_check"):
        pass
        
    #초기 state 설정
    WIP = 0
    ws_state = State_CNN("WS_json")
    car_state = State_CNN("Car_json")
    state = State(0,0,ws_state,car_state)
    #print(f'WS:{state.ws_state}Car:{state.car_state}')
    
    print(f"State {i}:")
    for j in range(action_count):
        #Action 선택
        #ws_action = action_dic[action[j]][0]
        car_action = action_dic[action[j]][1]
        
        if j == 0:
            reward = 0
        
        state_t = {'State_number' : state.state_num,
            'WIP': state.WIP,
            'ws_state': state.ws_state,
            'car_state': state.car_state,
            'action' : car_action,
            'reward' : reward}
        
        state_data.append(state_t)
        
        #시뮬레이션 진행
        plantsim.set_value("Rule_select",False)
        plantsim.set_value("Path_Rule", car_action)
        plantsim.start_simulation()
        print(f"  Action {j}: WS Action - {0}, path Action - {car_action}")
        while not plantsim.get_value("Rule_select"):
            if plantsim.get_value("Terminate"):
                break
            pass
        
        #다음 상태 확인
        WIP = plantsim.get_value("WIP")
        ws_state = round(State_CNN("WS_json"),8)
        car_state = round(State_CNN("Car_json"),8)
        next_state = State(j+1,WIP,ws_state,car_state)
              
        #결과 확인
        if j+1 == action_count:
            if WIP != 0:
                reward = plantsim.get_value("Reward")
                makespan = {"Episode" : i+1, "Makespan" : "Not_finish"}
                c_max_list.append(999999)
            else:
                reward = 100000000 - plantsim.get_value("Cmax")*100
                makespan = {"Episode" : i+1, "Makespan" : plantsim.get_value("Cmax")}
                c_max_list.append(plantsim.get_value("Cmax"))
        else:
            reward = plantsim.get_value("Reward")
        
        #다음 액션 확인
        if j+1 < action_count: 
            car_next_action = action_dic[action[j+1]][1]
            
        #모델 학습
        agent.learn(state, car_action, action_count, reward, next_state, car_next_action)
        
        #State 갱신
        state = next_state

    all_state.append(pd.DataFrame(state_data))

end_time = time.time()
spent_time = end_time - start_time
min_makespan = min(c_max_list)
best_action_index = c_max_list.index(min_makespan)+1

time_calculate("spent_time",spent_time)
time_calculate("min_makespan",min_makespan)
print(f'Best_Scenario : {all_cases[best_action_index]}')

# Q-테이블 데이터를 준비합니다.
q_table_data = []

# Q-테이블 항목을 순회하며 데이터를 리스트에 추가합니다.
for key, value in agent.q_table.items():
    state_info, action = key  # key를 state_info와 action으로 분리
    state_num, WIP, ws_state, car_state = state_info
    q_table_data.append({
        "State_num": state_num,
        "WIP": WIP,
        "WS_state": ws_state,
        "Car_state": car_state,
        "Action": action,
        "Q-Value": value
    })

q_table_data = pd.DataFrame(q_table_data)
q_table_data.to_excel("C:/Users/leewa/Desktop/Q-table/Episode_table.xlsx")
c_max_list = pd.DataFrame(c_max_list)
c_max_list.to_excel("C:/Users/leewa/Desktop/Q-table/c_max_list.xlsx")

with pd.ExcelWriter("C:/Users/leewa/Desktop/Q-table/State_table.xlsx") as writer:
    for idx, df in enumerate(all_state):
        sheet_name = f'Episode{idx + 1}'  # 각 시트의 이름을 'Sheet1', 'Sheet2', ...로 설정
        df.to_excel(writer, sheet_name=sheet_name, index=False)
