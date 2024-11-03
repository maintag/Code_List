import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plantsim.plantsim import Plantsim
from plantsim.table import Table
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    torch.manual_seed(42)
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

# 모델 연결
path_1 ="C:/Users/leewa/Desktop/EX.spp"
plantsim = Plantsim(version='23.2', visible ='True', license_type='educational')
plantsim.load_model(path_1)
plantsim.set_path_context('.models.model')

#데이터 추출
DT = Table(plantsim, "DataTable").rows_coldict

#각 값을 구분할 수 있는 value로 변경
word_to_index = {}
index = 0
for sequence in DT:
    for word in sequence:
        if word not in word_to_index:
            word_to_index[word] = index
            index += 1
    
#입력 데이터를 정수로 변경 -> Tensor
indexed_data = [[word_to_index[word] for word in sequence] for sequence in DT]
tensor_data = torch.tensor(indexed_data,dtype=torch.float32)
    
# Tersor shape 확인
height, width = tensor_data.shape[0], tensor_data.shape[1]
state_tensor = tensor_data.unsqueeze(0).unsqueeze(0)

model = CNN_Model(height, width)
state_feature = model(state_tensor)
ws_state = state_feature.detach().numpy()
print(ws_state)