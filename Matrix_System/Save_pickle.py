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

pickle_path = 'C:/Users/Whan Lee/Desktop/Matrix System/Code/Pickle/'

def time_calculate(spent):
# 시간, 분, 초로 변환
    hours = int(spent // 3600)
    minutes = int((spent % 3600) // 60)
    seconds = spent % 60
    print(f"종료시간 : {hours:02}:{minutes:02}:{seconds:06.2f}")

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

def simulation_module(simulation_path):
    plantsim = Plantsim(version='23.2', license_type='Educational', visible = True)
    plantsim.load_model(simulation_path)
    plantsim.set_path_context('.models.model')
    plantsim.set_event_controller()
    return plantsim

def check_makespan(model,action_dic):
    c_max_list = []
    for i in range(len(action_dic)):
        model.reset_simulation()
        done = model.get_value("Terminate")
        ws_action, car_action = action_dic[i]
        print(ws_action, car_action)
        while not done:
            model.set_value("Path_Rule", ws_action)
            model.set_value("Nextwork_Rule", car_action)
            model.set_value("Rule_select", False)
            model.start_simulation()
            while not model.get_value("Rule_select"):
                done = model.get_value("Terminate")
                if done:
                    break
                pass
        c_max_list.append(model.get_value("Cmax"))
    return c_max_list

def State_CNN(table_name,model):
    torch.manual_seed(42)
    #데이터 추출  
    state_json = json.loads(model.get_value(table_name))
    converted_list = []
    
    for item in state_json:
        numeric_values = [float(value) for value in item.values()]
        converted_list.append(numeric_values)
        
    tensor_data = torch.tensor(converted_list, dtype=torch.float32)
    state_tensor = tensor_data.unsqueeze(0).unsqueeze(0)
    height, width = tensor_data.shape[0], tensor_data.shape[1]
    
    feature_model = CNN_Model(height, width)
    feature = feature_model(state_tensor)
    feature = feature.detach()
    return feature

#path_1 ="C:/Users/Whan Lee/Desktop/Matrix System/Simulation/GM_Matrix_model_240819.spp"
#action_dic = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
#c_max_list = []

#plantsim = simulation_module(path_1)
#single_rule_makespan = check_makespan(plantsim,action_dic,c_max_list)
#print(single_rule_makespan)

def count_def(pickle_path):
    # 'def ' 키워드의 개수를 세어 반환
    code = """def time_calculate(spent):
    # 시간, 분, 초로 변환
        hours = int(spent // 3600)
        minutes = int((spent % 3600) // 60)
        seconds = spent % 60
        print(f"종료시간 : {hours:02}:{minutes:02}:{seconds:06.2f}")

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

    def simulation_module(simulation_path):
        plantsim = Plantsim(version='23.2', license_type='Educational', visible = True)
        plantsim.load_model(simulation_path)
        plantsim.set_path_context('.models.model')
        plantsim.set_event_controller()
        return plantsim

    def check_makespan(model,action_dic,c_max_list):
        c_max_list = []
        for i in range(len(action_dic)):
            model.reset_simulation()
            done = model.get_value("Terminate")
            ws_action, car_action = action_dic[i]
            print(ws_action, car_action)
            while not done:
                model.set_value("Path_Rule", ws_action)
                model.set_value("Nextwork_Rule", car_action)
                model.set_value("Rule_select", False)
                model.start_simulation()
                while not model.get_value("Rule_select"):
                    done = model.get_value("Terminate")
                    if done:
                        break
                    pass
            c_max_list.append(model.get_value("Cmax"))
        return c_max_list
    
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
        return feature"""
    
    def_count = code.count('def ')
    function_names = re.findall(r'def (\w+)\(', code)
    
    with open(pickle_path+'function_names.txt', 'w') as file:
        file.write('\n'.join(function_names))
    
    return def_count, function_names

def_count, function_names = count_def(pickle_path)

for i in range(def_count):
    function_obj = globals()[function_names[i]]
    with open(pickle_path + function_names[i] + '.pkl', 'wb') as f:
        dill.dump(function_obj, f)