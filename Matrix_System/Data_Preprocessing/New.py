# 기본 설정 라이브러리
import re, json, time, math, dill, random, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# PlantSimulation API 라이브러리
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

# 기준 정보 및 학습 환경 경로 설정
pickle_path = 'C:/Users/Whan Lee/Desktop/Matrix System/Code/Pickle/'
simulation_path = "C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240827.spp"

# 정의된 피클 파일 로드(time_calculate, c_max_chart,moving_average_chart, simulation_module, check_makespan, State_CNN)
with open(pickle_path+'function_names.txt', 'r') as f:
    function_names = f.read().splitlines()
    for function_name in function_names:
        with open(pickle_path + function_name + '.pkl', 'rb') as f:
            globals()[function_name] = dill.load(f)

# GPU 환경구축
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# State data 피처링(WIP, WS, Car)
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

# RolloutBuffer 정의
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


# ActorCritic 네트워크 정의
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        self.device = device
        # Actor network - 행동 확률 출력 및 선택
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network - 상태 가치 평가
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.to(self.device)  # 모델을 설정된 디바이스로 이동

        # 네트워크 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def act(self, state):
        # 상태 텐서 확인
        state = state.clone().detach().to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Actor 네트워크를 통해 행동 확률 계산
        action_probs = self.actor(state)
        action_probs = action_probs + torch.rand_like(action_probs) * 0.01
        
        # Categorical 분포 생성
        dist = torch.distributions.Categorical(action_probs)
        
        # Categorical 분포에서 행동 샘플링 
        action = dist.sample()
        
        # 선택된 액션의 로그 확률 계산
        action_logprob = dist.log_prob(action)
        
        # Critic 네트워크에서 상태 값 계산
        state_value = self.critic(state).squeeze()  # 배치 차원 제거

        return action.item(), action_logprob, state_value.item()

    def evaluate(self, state, action):
        # 상태와 액션을 장치에 맞게 이동 및 텐서 확인
        state = state.clone().detach().to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device) if not isinstance(action, torch.Tensor) else action
        
        # Actor 네트워크에서 행동 확률 계산
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
            
        # 주어진 행동에 대한 로그 확률 및 엔트로피 계산
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        # Critic 네트워크에서 상태 값 평가 (스칼라로 변환)
        state_value = self.critic(state).squeeze()

        return action_logprobs, state_value, dist_entropy
    
class PPO:
    def __init__(self, state_dim, action_dim, max_training_steps, device):
        # 총 에피소드 수
        self.num_episodes = 10000  # 에이전트가 환경과 상호작용하는 최대 에피소드 수
        
        # 할인 인자 (Discount factor) & GAE 람다
        self.gamma = 0.999  # 미래 보상에 대한 할인율로, 0에 가까울수록 미래 보상보다 즉각적인 보상에 가중치를 둠
        self.lamda = 0.999  # GAE에서 사용되는 파라미터로, 1에 가까울수록 더 긴 시간 범위에 대한 보상 예측
        
        # Cliping 인자 & 반복 에포크 수
        self.eps_clip = 0.9  # PPO의 클리핑 파라미터로, 정책이 너무 많이 변하지 않도록 제한함. 작은 값일수록 더 보수적인 업데이트
        self.K_epochs = 100  # 매 업데이트마다 손실 함수를 최소화하기 위해 반복하는 횟수. 큰 값일수록 업데이트가 천천히 진행됨
        
        # Critic 손실(c1), 엔트로피 보너스(c2) 가중치
        self.c1 = 0.5  # Critic 네트워크의 손실(Loss)에 대한 가중치로, Critic 손실에 얼마나 비중을 둘지 결정
        self.c2 = 0.1  # 정책의 엔트로피 보너스에 대한 가중치로, 탐험을 장려하는 역할을 함. 값이 클수록 탐험이 많아짐
        
        # Learning rate
        self.lr_actor = 0.0005
        self.lr_critic =0.001
        
        # Update 주기
        self.update_timestep = max_training_steps * 10  # 주어진 max_training_steps에 기반한 업데이트 간격
        self.time_step = 0  # 현재 시간 스텝 초기화
        
        # RolloutBuffer 네트워크
        self.buffer = RolloutBuffer()
        
        # ActorCritic 네트워크 (GPU 활용)
        self.device = device
        self.policy = ActorCritic(state_dim, action_dim, self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, self.device)
        
        # Optimizer 설정
        self.optimizer = optim.Adam([{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                                     {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}])
        
        # 이전 정책을 저장할 네트워크 동기화
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 손실 함수
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # 상태를 self.device로 이동
        state = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32).to(self.device)
    
        
        # 기울기 계산 생략 
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state)
        
        return action, action_logprob, state_value
    
    def store_transition(self, state, action, action_logprob, state_value, reward, is_terminal):
        """환경에서 얻은 보상과 종료 상태를 저장"""
        # 학습 정보 버퍼에 저장
        self.buffer.states.append(state.cpu())
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob.cpu())
        self.buffer.state_values.append(state_value)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)
    
    def compute_discounted_rewards(self):
        """할인된 보상을 계산하는 함수"""
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return torch.FloatTensor(rewards)
        
    def update(self):
        # 버퍼에서 데이터를 가져와 텐서로 변환
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(self.device)
        rewards = self.compute_discounted_rewards().to(self.device)
        old_state_values = torch.FloatTensor(self.buffer.state_values).to(self.device)

       # 크기 불일치 체크
        assert rewards.shape[0] == old_state_values.shape[0], "rewards와 old_state_values의 크기가 일치하지 않습니다."
        
        # GAE(Generalized Advantage Estimation) 계산
        advantages = rewards - old_state_values

        # K번 반복하여 정책과 가치 함수 업데이트 (PPO 클리핑 적용)
        for _ in range(self.K_epochs):
            # 현재 정책에 따른 로그 확률, 상태 가치, 엔트로피 계산
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            # 중요도 비율 계산 (new_probs / old_probs)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # PPO 클리핑 손실 계산
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 가치 함수 손실 (Critic)
            value_loss = self.c1 * F.mse_loss(state_values, rewards)

            # 엔트로피 보너스 (새로운 액션 탐험)
            entropy_loss = -self.c2 * dist_entropy.mean()

            # 총 손실
            loss = policy_loss + value_loss + entropy_loss

            # 옵티마이저 업데이트
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()

        # 새로운 정책을 old 정책으로 복사
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 버퍼 초기화
        self.buffer.clear()
        
# 상태와 액션 정보
state_dic = {0: "WIP", 1: "WS", 2: "Car"}
action_dic = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}

# 액션 선택 주기 설정
time_step = 9000

# 시뮬레이션 설정
plantsim = simulation_module(simulation_path, time_step)

# 에피소드 별 makespan 기록(단일 규칙보다 오래 걸릴시 60000으로 설정)
c_max_list = []
episode_rewards = []
min_cmax = 45000 

# 학습 시작
Start_time = time.time()

# 에피소드 내 최대 범위 설정(단일 규칙 중 최소 값 기준)
max_training_steps = math.ceil(min_cmax / time_step)

# PPO 에이전트 설정
agent = PPO(len(state_dic), len(action_dic), max_training_steps, device)
for i in range(agent.num_episodes):
    episode_reward = 0
    
    # 초기 state 확인
    plantsim.reset_simulation()
    WIP = torch.tensor(plantsim.get_value("WIP"))
    next_state = torch.tensor([WIP.item(), State_CNN("WS_json",plantsim).item(), State_CNN("Car_json",plantsim).item()])
    done = plantsim.get_value("Terminate")
    print(f"Episode {i}")
    
    for j in range(max_training_steps):
        state = next_state
        action, action_logprob, state_value = agent.select_action(state)
        
        print(action)
        
        ws_action, car_action = action_dic[action]
        
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
            print(c_max)
            if c_max <= min_cmax:
                reward = (100000 - c_max) * 0.01  # 보상 스케일 조정
            else:
                reward = 10
            break
        else:
            c_max, reward= 100000, plantsim.get_value("Reward")* 0.1
            
        episode_reward += reward
        
        next_state = torch.tensor([plantsim.get_value("WIP"), State_CNN("WS_json",plantsim).item(), State_CNN("Car_json",plantsim).item()])
        
        # Reward, Terminal 정보 저장 
        agent.store_transition(state, action, action_logprob, state_value, reward, done)
        
        # PPO 정책 업데이트
        agent.time_step +=1
        if agent.time_step > agent.update_timestep:
            agent.update()
            agent.time_step = 0  # 업데이트 후 다시 0으로 초기화
    
    # 에피소드 종료 시 보상 기록
    episode_rewards.append(episode_reward)
    c_max_list.append(c_max)
    
    if i % 1000 == 0 or i == agent.num_episodes-1:
        c_max_chart(c_max_list)
        moving_average_chart(c_max_list,100)
        
        # 주기적으로 보상 그래프를 업데이트
    if i % 100 == 0:
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.show()
End_time = time.time()
Spent_time = End_time - Start_time
time_calculate(Spent_time)