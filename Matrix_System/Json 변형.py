a =[[613.6845167064821, 628.0, 628.0, 628.0, 0.0],
 [521.5581829178136, 618.0, 628.0, 628.0, 0.0],
 [808.5442646942074, 1201.0, 1201.0, 1203.0, 1203.0],
 [347.04975153885516, 2021.0, 2010.0, 1392.0, 820.0],
 [492.5810565254087, 628.0, 618.0, 628.0, 0.0],
 [90.04146833331197, 384.0, 0.0, 0.0, 0.0],
 [185.98190178916593, 1201.0, 1201.0, 1201.0, 575.0],
 [123.10372270504104, 628.0, 628.0, 628.0, 0.0],
 [347.30346353367077, 626.0, 626.0, 628.0, 628.0],
 [779.7771087794163, 1395.0, 1395.0, 1395.0, 1392.0],
 [43.17633434421532, 618.0, 626.0, 628.0, 628.0],
 [411.65489808424536, 603.0, 575.0, 1203.0, 572.0],
 [103.56676804746166, 0.0, 0.0, 0.0, 0.0],
 [97.4328946333626, 575.0, 572.0, 572.0, 0.0],
 [465.8956137085761, 820.0, 820.0, 820.0, 1216.0],
 [473.8131210861611, 0.0, 0.0, 0.0, 0.0],
 [369.76017432069693, 628.0, 1460.0, 628.0, 643.0],
 [1111.8998038677419, 0.0, 0.0, 0.0, 0.0],
 [653.6293556401461, 1392.0, 824.0, 1395.0, 820.0]]

import itertools
import pandas as pd
import numpy as np
import torch.optim as optim
import random
import json

from collections import deque
from plantsim.plantsim import Plantsim 
from plantsim.table import Table

import matplotlib.pyplot as plt
import torch
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
    
def State_CNN(table_name):
    
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

path_1 ="C:/Users/leewa/Desktop/test.spp"
#path_1 = "D:/Python/01_Project/GM_Matrix/GM_Matrix_model_240723.spp"

plantsim = Plantsim(version='23.2', license_type='Educational', visible = True)
plantsim.load_model(path_1)
plantsim.set_path_context('.models.model')
plantsim.set_event_controller()


ws_state = State_CNN("J")
print(ws_state)
