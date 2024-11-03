import re, json, time, math, dill, random, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from plantsim.plantsim import Plantsim 
from plantsim.table import Table

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
from torch.distributions import Categorical 

# 정의된 피클파일 로드(time_calculate, c_max_chart,moving_average_chart, simulation_module, check_makespan, State_CNN)
pickle_path = 'C:/Users/Whan Lee/Desktop/Matrix System/Code/Pickle/'
simulation_path = "C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240917.spp"

with open(pickle_path+'function_names.txt', 'r') as f:
    function_names = f.read().splitlines()
    for function_name in function_names:
        with open(pickle_path + function_name + '.pkl', 'rb') as f:
            globals()[function_name] = dill.load(f)

# GPU 환경 구축
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

# Featuring of state data(WIP, WS, Car)
class CNN_Model(nn.Module):
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
    
# Buffer class for storing policy and value function data
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
        
# Relu : 이산적인 액션 선택(tanh<-무의미)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor 네트워크
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1))
        
        # Critic 네트워크
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1))
        
        # 네트워크 가중치 초기화 (Xavier 초기화 또는 0으로 초기화)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Actor 네트워크의 가중치 초기화 (Xavier 초기화)
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # Xavier 초기화
                init.constant_(layer.bias, 0)  # 편향은 0으로 초기화

        # Critic 네트워크의 가중치 초기화
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.1)  # 편향 값을 조금 더 크게 초기화
                
    # 행동 선택    
    def act(self, state):
        with torch.no_grad():  # 기울기 계산 비활성화(불필요한 계산 감소)  
            # Actor 네트워크로부터 행동 확률 계산
            action_probs = self.actor(state)
            #print(f"Action probabilities: {action_probs}")  # 행동 확률 출력
            dist = Categorical(action_probs)
            
            # Categorical 분포에서 행동 샘플링
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
            # Critic 네트워크로부터 상태 가치를 계산 (차원 축소)
            state_value = self.critic(state).squeeze(-1)
    
        return action.item(), action_logprob, state_value
    
    # 정책 평가 및 손실 계산
    def evaluate(self, state, action):
        action =  torch.tensor(action)
        # Actor 네트워크로부터 행동 확률 계산
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        # 행동 확률과 엔트로피 계산
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Critic 네트워크로부터 상태 가치를 계산
        state_values = self.critic(state).squeeze(-1)
        return action_logprobs, state_values.squeeze(), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim,max_training_steps):
        self.num_episodes = 10000
        self.gamma = 0.999
        self.lamda = 0.999
        self.eps_clip = 0.2
        self.K_epochs = 100
        self.c1 = 0.5  # Critic 손실 가중치
        self.c2 = 0.1  # 엔트로피 보너스 가중치
        self.lr_actor = 0.0003
        self.lr_critic =0.001
        self.update_timestep =  max_training_steps * 10
        self.time_step = 0
        
        # RolloutBuffer 네트워크
        self.buffer = RolloutBuffer()    
        
        # ActorCritic 네트워크
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam([{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                                     {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}])
        
        # 이전 정책을 저장할 네트워크
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 손실 함수
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        
        # 이산적인 행동 선택
        with torch.no_grad():
            state = state.float()
            action, action_logprob, state_value = self.policy_old.act(state)
            print(f"Action probabilities: {action_logprob}")
        return action, action_logprob, state_value
    
    def update(self):
        # Buffer에서 데이터 로드
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
    
        # 보상 정규화
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
    
        # 리스트를 텐서로 변환
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.tensor(self.buffer.actions).detach()
        old_logprobs = torch.tensor(self.buffer.logprobs).detach()
        old_state_values = torch.stack(self.buffer.state_values).detach()
    
        # PPO update for K epochs
        for _ in range(self.K_epochs):
            # 평가 네트워크에서 새로 계산된 값
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Advantage 계산 (A = R - V)
            advantages = rewards - old_state_values.squeeze()
    
            # 확률 비율 (new / old)
            ratios = torch.exp(logprobs - old_logprobs)
    
            # PPO 클리핑 손실
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
    
            # Critic 손실
            loss_critic = agent.MseLoss(state_values, rewards)
    
            # 전체 손실
            total_loss = loss_actor + self.c1 * loss_critic - self.c2 * dist_entropy
            total_loss = total_loss.mean()
    
            # 정책 업데이트
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
        # 이전 정책 네트워크 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Buffer 초기화
        self.buffer.clear()

    def store_transition(self, state, action, action_logprob, state_value, reward, done):
        # 버퍼에 저장
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    
action_dic = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
state_dic = {0: "WIP", 1: "WS", 2: "Car"}

time_step = 9000
plantsim = simulation_module(simulation_path, time_step)
c_max_list = []

'''
# 단일 우선순위 규칙 기반의 Makespan 확인
single_rule_makespan = check_makespan(plantsim,action_dic)
single_rule_makespan_list = single_rule_makespan[0]
min_cmax = single_rule_makespan[1]
print(f'Single rule makespan_list : {single_rule_makespan_list}')
print(f'Rule Number : {single_rule_makespan_list.index(min_cmax)} / Min Makespan : {min_cmax}')
'''

min_cmax = 45000

# 학습 시작
Start_time = time.time()
update_timestep = 0
max_training_steps = math.ceil(min_cmax / time_step)

agent = PPO(len(state_dic), len(action_dic),max_training_steps)
for i in range(agent.num_episodes):
    # 초기 state 확인
    plantsim.reset_simulation()
    WIP = plantsim.get_value("WIP")/75
    next_state = torch.tensor([WIP/75, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")])
    #next_state = torch.tensor([WIP.item(), State_CNN("WS_json",plantsim).item(), State_CNN("Car_json",plantsim).item()])
    done = plantsim.get_value("Terminate")
    print(f"Episode {i}")
    action_list_per_episode = []
    action_prob_list_per_episode = []
    
    for j in range(max_training_steps):
        state = next_state
        reward = 0
        action, action_logprob = agent.select_action(state)
        
        action_list_per_episode.append(action.item())
        action_prob_list_per_episode.append(action_logprob)
        
        ws_action, car_action = action_dic[action.item()]
        
        # 시뮬레이션 내 액션 적용
        plantsim.set_value("Path_Rule", ws_action)
        plantsim.set_value("Nextwork_Rule", car_action)
        plantsim.set_value("Rule_select", False)
        plantsim.start_simulation()
        
        # 학습 종료 확인
        while not plantsim.get_value("Rule_select"):
            done = plantsim.get_value("Terminate")
            if done:
                break
            pass
        
        # 적용 리워드 확인
        if done:
            c_max = plantsim.get_value("Cmax")
            if c_max <= min_cmax: 
                reward = (100000 - c_max) * 0.01  # 보상 스케일 조정
            else:
                reward = 0
            break
        else:
            c_max, reward= 100000, plantsim.get_value("Reward_Progress")
        
        WIP = plantsim.get_value("WIP")/75
        next_state = torch.tensor([WIP/75, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")]) 
        # Reward 저장 
        agent.store_transition(state, action, action_logprob, state_value, reward, done)
        
        # PPO 정책 업데이트
        agent.time_step +=1
        if agent.time_step % agent.update_timestep == 0:
            agent.update()
            agent.time_step = 0  # 업데이트 후 다시 0으로 초기화
            print(action_list_per_episode)
            print(action_prob_list_per_episode)
            
    c_max_list.append(c_max)
    
    
    if i % 1000 == 0 or i == agent.num_episodes-1:
        c_max_chart(c_max_list)
        moving_average_chart(c_max_list,100)

End_time = time.time()
Spent_time = End_time - Start_time
time_calculate(Spent_time)