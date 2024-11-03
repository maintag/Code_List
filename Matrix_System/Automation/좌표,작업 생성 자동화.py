import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill
import math

n_WS = 19

def generate_points_grid(n_WS, num_lines, x_gap, y_gap):
    # 포인트 그리드 생성
    points_per_line = (n_WS + num_lines - 1) // num_lines
    data = {'x-axis': [], 'y-axis': []}
    row_labels = [f'WS_{i+1}' for i in range(n_WS)]
    for i in range(num_lines):
        for j in range(points_per_line):
            if len(data['x-axis']) >= n_WS:
                break
            data['x-axis'].append(j * x_gap)
            data['y-axis'].append(i * y_gap)

    # 데이터 프레임 생성
    df = pd.DataFrame(data, index=row_labels)
    return df

def plot_points(df):
    # 포인트 그리드 시각화
    plt.figure(figsize=(8, 8))
    for index, row in df.iterrows():
        plt.scatter(row['x-axis'], row['y-axis'], color='blue', label=index)

    plt.xlim(-10, df['x-axis'].max() + 10)
    plt.ylim(-10, df['y-axis'].max() + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def possible_operation(n_WS=19, min_proc=4, max_proc=7):

    # 컬럼 명 생성
    columns = [f'WS_{i+1}' for i in range(n_WS)]
    
    # row 명 설정
    rows = ['VIN', 'Doors off', 'Weatherstrips', 'MBWH', 'Sunroof', 'RRAB',
            'Headliner', 'Cockpit', 'Seat Belts', 'Console', 'Carpet',
            'Interior hard trim', 'Seats', '1/4 glass (W/S & backlite)',
            'Lower exterior trim molding', 'Upper exterior trim molding',
            'Headlamps', 'Front & Rear Fascias', 'Doors On']
    # 데이터프레임 생성
    df = pd.DataFrame(0, index=rows, columns=columns)
    
    # 각 컬럼에 대해 4에서 7개의 row를 랜덤하게 선택하여 1로 설정
    for column in columns:
        selected_rows = np.random.choice(rows, 
                                         size=np.random.randint(min_proc, max_proc+1), 
                                         replace=False)
        df.loc[selected_rows, column] = 1
        
    # 모든 행에 최소 2개의 1이 있는지 확인하고, 아닌 경우 추가 조정
    for index, row in df.iterrows():
        while row.sum() < 2:
            col = np.random.choice(df.columns)
            df.loc[index, col] = 1
    
    df_transposed = df.T
    
    return df_transposed

# 좌표 생성
n_WS=15

PositionWS = generate_points_grid(n_WS, num_lines=int(math.sqrt(n_WS)), x_gap=30, y_gap=30)
plot_map = plot_points(PositionWS)
print(PositionWS)
# 데이터프레임 생성

OperationWS = possible_operation(n_WS, min_proc=6, max_proc=9)

PositionWS.to_excel("C:/Users/Whan Lee/Desktop/PositionWS.xlsx")
OperationWS.to_excel("C:/Users/Whan Lee/Desktop/OperationWS.xlsx")


"""
with open('C:/Users/leewa/Desktop/generate_points_grid.pkl', 'wb') as file1:
    dill.dump(generate_points_grid, file1)
with open('C:/Users/leewa/Desktop/plot_points.pkl', 'wb') as file2:
    dill.dump(plot_points, file2)
with open('C:/Users/leewa/Desktop/possible_operation.pkl', 'wb') as file3:
    dill.dump(possible_operation, file3)




# 함수를 피클 파일로 저장
with open('C:/Users/leewa/Desktop/저장.pkl', 'wb') as file:
    dill.dump(generate_points_grid, file)
    
# 피클 파일로부터 함수 로드
with open('C:/Users/leewa/Desktop/저장.pkl', 'rb') as file:
    loaded_function = dill.load(file)
"""
