import re, json, time, math, dill, random, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from plantsim.plantsim import Plantsim 
from plantsim.table import Table

# 신경망 라이브러리
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
from torch.distributions import Categorical 
from torch.utils.data import TensorDataset, DataLoader

# 정의된 피클파일 로드(time_calculate, c_max_chart,moving_average_chart, simulation_module, check_makespan, State_CNN)
pickle_path = 'C:/Users/Whan Lee/Desktop/Matrix System/Code/Pickle/'
simulation_path = "C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240917.spp"

with open(pickle_path+'function_names.txt', 'r') as f:
    function_names = f.read().splitlines()
    for function_name in function_names:
        with open(pickle_path + function_name + '.pkl', 'rb') as f:
            globals()[function_name] = dill.load(f)

# device 설정 (GPU가 있으면 CUDA, 없으면 CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): 
    torch.cuda.empty_cache()

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_logprob = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def store(self, state, action, action_logprob, state_value, reward, is_terminal):
        """PPO에 필요한 데이터를 저장하는 함수"""
        self.states.append(state)
        self.actions.append(action)
        self.action_logprob.append(action_logprob)
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def get(self):
        """저장된 데이터를 반환하고 텐서로 변환"""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.action_logprob),
            torch.stack(self.state_values),
            torch.tensor(self.rewards),
            torch.tensor(self.is_terminals))
    
    def clear(self):
        """저장된 데이터를 초기화"""
        del self.states[:]
        del self.actions[:]
        del self.action_logprob[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
    def size(self):
        return len(self.states)
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Actor 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1))
        
        # Critic 네트워크
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
    def select_action(self, state, train):
        if train:  # 학습 모드
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            state_value = self.critic(state).squeeze(-1)
        
        else: # 평가 모드
            with torch.no_grad():
                action_probs = self.actor(state)
                dist = Categorical(action_probs)
                action = dist.sample()
                state_value = self.critic(state).squeeze(-1)
        
        action_logprob = dist.log_prob(action)
        
        return action, action_probs, action_logprob, state_value
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state).squeeze(-1)
        
        return action_logprobs, state_values, entropy

class PPO:
    def __init__(self, state_dim, action_dim, max_training_steps):
        self.num_episodes = 20000
        self.gamma = 0.99
        self.lamda = 0.95
        self.eps_clip = 0.3
        self.c1 = 0.2 # Critic 손실 가중치
        self.c2 = 0.05  # 엔트로피 보너스 가중치
        self.lr_actor = 0.002
        self.lr_critic = 0.0005
        self.batch_size = 256
        self.K_epochs = 4
        self.n_steps = 500
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RolloutBuffer 네트워크
        self.buffer = RolloutBuffer()
        
        # ActorCritic 네트워크
        self.policy = ActorCritic(state_dim, action_dim, 1024).to(self.device)
        self.optimizer = optim.Adam([{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                                     {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}])

    def select_action(self, state, train):
        # policy의 select_action을 사용하여 행동 선택
        action, action_probs, action_logprob, state_value = self.policy.select_action(state, train)
        return action, action_probs, action_logprob, state_value
    
    def compute_returns_and_advantages(self, rewards, state_values, is_terminals, device):
        """GAE를 이용한 리턴 및 Advantage 계산"""
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        terminal_weight = 10  # done = True일 때 적용할 가중치
    
        for t in reversed(range(len(rewards))):
            # bool 값을 float 또는 int로 변환하여 산술 연산이 가능하도록 수정
            is_terminal = float(is_terminals[t])  # 또는 int(is_terminals[t])
            
            if is_terminal:
                next_value = 0
                gae = 0
            
            # Temporal Difference Residual (delta)
            #if is_terminal:
             #   delta = (rewards[t] * terminal_weight) + self.gamma * next_value * (1 - is_terminal) - state_values[t]
            #else:
             #   delta = rewards[t] + self.gamma * next_value * (1 - is_terminal) - state_values[t]
            # Temporal Difference Residual (delta)
            if is_terminal:
                delta = (rewards[t] * terminal_weight) + self.gamma * next_value * (1 - is_terminal) - state_values[t]
            else:
                delta = rewards[t] + self.gamma * next_value * (1 - is_terminal) - state_values[t]
                
            # GAE를 통한 Advantage 계산
            gae = delta + self.gamma * self.lamda * gae * (1 - is_terminal)
            advantages.insert(0, gae)
            
            # 리턴 값 계산 (최종 상태에 대한 리턴을 강화)
            if is_terminal:
                returns.insert(0, gae * terminal_weight + state_values[t])
            else:
                returns.insert(0, gae + state_values[t])
    
            # 리턴 값 계산
            #next_value = state_values[t]
            #returns.insert(0, gae + state_values[t])
    
        returns = torch.tensor(returns).to(device)
        advantages = torch.tensor(advantages).to(device)
    
        return returns, advantages

    
    def update(self):
        """PPO 정책 및 가치 네트워크 업데이트"""
        # Buffer에서 데이터 가져오기
        states, actions, old_logprobs, state_values, rewards, is_terminals = self.buffer.get()
        
        # 할인된 리턴 및 Advantage 계산
        returns, advantages = self.compute_returns_and_advantages(rewards, state_values, is_terminals, states.device)
        
        # 데이터를 섞기 위해 랜덤 순서 생성
        perm = torch.randperm(len(states))
        
        # 데이터를 랜덤 순서로 섞기
        states = states[perm]
        actions = actions[perm]
        old_logprobs = old_logprobs[perm]
        state_values = state_values[perm]
        returns = returns[perm]
        advantages = advantages[perm]
        
        # PPO 업데이트 (여러 에포크 동안 진행)
        for _ in range(self.K_epochs):
            # 버퍼에서 미니배치로 샘플링
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                batch_old_logprobs = old_logprobs[i:i+self.batch_size]
                batch_returns = returns[i:i+self.batch_size]
                batch_advantages = advantages[i:i+self.batch_size]

                # 현재 정책 평가 (새로운 로그 확률 및 상태 가치 계산)
                new_logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 확률 비율 계산 (ratio = exp(new_logprobs - old_logprobs))
                ratios = torch.exp(new_logprobs - batch_old_logprobs.detach())

                # 클리핑된 손실 함수
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic 손실 (리턴과 상태 가치의 차이)
                critic_loss = nn.MSELoss()(state_values, batch_returns)

                # 엔트로피 보너스 (정책의 탐험성 증가)
                loss = actor_loss + self.c1 * critic_loss - self.c2 * dist_entropy.mean()

                # 최종 손실에 대해 역전파 및 파라미터 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        self.buffer.clear()


action_dic = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
state_dic = {0: "WIP", 1: "WS", 2: "Car"}

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

# 학습 시작
Start_time = time.time()
update_timestep = 0
max_training_steps = math.ceil(min_cmax / time_step)

agent = PPO(len(state_dic), len(action_dic),max_training_steps)
for i in range(agent.num_episodes):
    # 초기 state 확인
    plantsim.reset_simulation()
    WIP = plantsim.get_value("WIP")/75
    next_state = torch.tensor([WIP, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")]).to(device)
    done = plantsim.get_value("Terminate")
    print(f"Episode {i}")
    action_list_per_episode = []
    action_probs_list = []
    
    #if i> 5000:
     #   agent.eps_clip *= 0.9996
    
    for j in range(max_training_steps):
        state = next_state
        reward = 0
        
        if i >= 9990:
            action, action_probs, action_logprob, state_value = agent.select_action(state, train= False)
        else:
            action, action_probs, action_logprob, state_value = agent.select_action(state, train= True)
        
        action_list_per_episode.append(action.item())
        action_probs_list.append(action_probs.squeeze(-1))
        
        
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
                reward = (50000 - c_max)*0.001 *((max_training_steps-j)*10) # 보상 스케일 조정
            else:
                reward = 5
            break
        else:
            c_max, reward= 45000, plantsim.get_value("Current_Progress")+ plantsim.get_value("Reward_Progress")
        
        WIP = plantsim.get_value("WIP")/75
        next_state = torch.tensor([WIP, plantsim.get_value("WorkingRate"), plantsim.get_value("Current_Progress")]).to(device)
        reward = torch.tensor(reward).to(device)
        is_terminal = torch.tensor(done).to(device)
        # Reward 저장 
        agent.buffer.store(state, action, action_logprob, state_value, reward, is_terminal)
        
        # PPO 정책 업데이트
        if agent.buffer.size() >= agent.n_steps:
            agent.update()
                    
    c_max_list.append(c_max)
    
    if c_max < min_makespan:
        min_makespan = c_max
        min_action = action_list_per_episode
    
    print(f'Makespan : {c_max} / Action_list_per_episode : {action_list_per_episode}')
    
    if i % agent.n_steps == 0 or i == agent.num_episodes-1:
        c_max_chart(c_max_list)
        moving_average_chart(c_max_list,agent.n_steps)

End_time = time.time()
Spent_time = End_time - Start_time
time_calculate(Spent_time)