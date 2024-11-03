import numpy as np
# 신경망 라이브러리
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical 


# RolloutBuffer 정의
class RolloutBuffer:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        
        self.batch_size = batch_size
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size) 

        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i+self.batch_size] for i in batch_start]
         
        return np.array(self.states), np.array(self.actions), \
               np.array(self.logprobs), np.array(self.rewards), \
               np.array(self.state_values), np.array(self.is_terminals), batches
 
               
# ActorCritic 네트워크 정의
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(ActorCritic, self).__init__()
        # Actor network - 행동 확률 출력 및 선택
        self.actor = nn.Sequential(
                        nn.Linear(state_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_size))
        
        # Critic network - 상태 가치 평가
        self.critic = nn.Sequential(
                        nn.Linear(state_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1))
        
        self.count = -1  # 초기화 식별 count
        self.device = device
        self.to(device)
        
# ActorCritic 네트워크 정의
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(ActorCritic, self).__init__()
        # Actor network - 행동 확률 출력 및 선택
        self.actor = nn.Sequential(
                        nn.Linear(state_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_size))
        
        # Critic network - 상태 가치 평가
        self.critic = nn.Sequential(
                        nn.Linear(state_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1))
        
        self.count = -1  # 초기화 식별 count
        self.device = device
        self.to(device)

    # Actor 신경망을 활용한 액션 선택
    def select_action(self, state):
        self.count += 1
        
        # 상태를 GPU로 이동
        state = state.to(self.device)
        
        if self.count == 0:
            # 처음 액션에는 동일한 확률을 부여
            action_probs_list = torch.ones(self.actor[-1].out_features).to(self.device) / self.actor[-1].out_features
        else:
            # 신경망을 통해 액션 확률을 계산
            action_probs_list = torch.softmax(self.actor(state), dim=-1)
        
        # Categorical 분포에서 액션을 샘플링
        dist = torch.distributions.Categorical(action_probs_list)
        action = dist.sample()  # 액션 샘플링
        
        # 선택된 액션의 로그 확률 계산
        action_prob = action_probs_list[action]
        
        # Critic 네트워크에서 상태 값 계산
        state_value = self.critic(state).squeeze()  # 배치 차원 제거
        return state, action, action_prob, action_probs_list, state_value
    
    # Evaluate 함수
    def evaluate(self, state, action):
        # 상태를 GPU로 이동
        state = state.to(self.device)

        # Actor 신경망으로 행동 확률 계산
        action_probs_list = torch.softmax(self.actor(state), dim=-1)
        
        # Categorical 분포 생성
        dist = Categorical(action_probs_list)
        
        # 선택된 행동의 로그 확률 계산
        log_probs = dist.log_prob(action)
        
        # 엔트로피 계산 (탐험성 증가를 위한 보너스)
        entropy = dist.entropy()
        
        # Critic 네트워크로 상태 가치 계산
        state_value = self.critic(state).squeeze()
        
        return log_probs, state_value, entropy

class PPO:
    def __init__(self, state_size, action_size, max_training_steps):
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
        self.c2 = 0.3  # 정책의 엔트로피 보너스에 대한 가중치로, 탐험을 장려하는 역할을 함. 값이 클수록 탐험이 많아짐
        
        # Learning rate
        self.lr_actor = 0.0005
        self.lr_critic =0.001
        
        # Update 주기
        self.update_timestep = max_training_steps * 10  # 주어진 max_training_steps에 기반한 업데이트 간격
        self.time_step = 0  # 현재 시간 스텝 초기화
        
        # RolloutBuffer 네트워크
        self.buffer = RolloutBuffer()
        self.state_size = state_size
        
        # GPU 환경 구축
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size, self.device)
        self.policy_old = ActorCritic(state_size, action_size, self.device)
        
        # Optimizer 설정
        self.optimizer = optim.Adam([{'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                                     {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}])
        
        # 이전 정책을 저장할 네트워크 동기화
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 손실 함수
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        """행동을 선택하고 로그 확률과 상태 값을 반환"""
        # 상태는 이미 ActorCritic 네트워크의 self.device에 맞춰 처리됨
        with torch.no_grad():
            state, action, action_prob, action_probs_list, state_value = self.policy_old.select_action(state)
            
        return state, action, action_prob, action_probs_list, state_value
    
    
    def store_transition(self, state, action, action_logprob, state_value, reward, done):
        """환경에서 얻은 보상과 종료 상태를 저장"""
        # 학습 정보 버퍼에 저장
        self.buffer.states.append(state.cpu())
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_prob.cpu())
        self.buffer.state_values.append(state_value.cpu())
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        
    
state = torch.tensor([12,0.12,0.22])
agent = PPO(3,9,100)

state, action, action_prob, action_probs_list, state_value = agent.select_action(state)

reward = 100
done = False

agent = PPO(3,9,100)
agent.select_action(state)