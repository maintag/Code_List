import random
import numpy as np
import random
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt

data =pd.read_csv('C:/Users/Whan Lee/Desktop/정리/다이텍/데이터/공정변수.csv',encoding='euc-kr')
df = pd.DataFrame(data)

x = df[['GUpSpeed1','GUpSpeed2','GUpSpeed3','GUpTemp3','GMaintain3']]
y = df[['K/S']]

# 'K/S' 열만 선택
y_column = 'K/S'
xy = df[[y_column]]  # 'K/S' 열만 포함된 데이터프레임 생성

# 산점도 그리기
#fig, ax = plt.subplots(figsize=(12, 6))
#ax.scatter(xy.index, xy[y_column])  # x축에 행 번호(index)를 사용, y축에 'K/S' 값 사용
#ax.set_xlabel('Dyeing Number')
#ax.set_ylabel('The Value of Residual Dye')
#ax.set_title('Scatter Plot of Residual dye')
#plt.show()


import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv('C:/Users/Whan Lee/Desktop/공정변수.csv',encoding='euc-kr')
x = data[['GUpSpeed1','GUpSpeed2','GUpSpeed3','GUpTemp3','GMaintain3']]
y = data['K/S']

# 모델 훈련
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(x, y)

# 컬럼 이름 변경
x.columns = ['UpSpeed1','UpSpeed2','UpSpeed3','UpTemp3','Maintain3']

# SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

# 특성 중요도 출력
shap.summary_plot(shap_values, x, plot_type='bar')



data =pd.read_csv('C:/Users/Whan Lee/Desktop/공정변수.csv',encoding='euc-kr')
x = data[['GUpSpeed1','GUpSpeed2','GUpSpeed3','GUpTemp3','GMaintain3']]
y = data['K/S']

# 컬럼 이름 변경
x.columns = ['UpSpeed1','UpSpeed2','UpSpeed3','UpTemp3','Maintain3']

# 모델 훈련
model = GradientBoostingRegressor(n_estimators=500, random_state=42)
model.fit(x, y)

# SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

# 특성 중요도 출력
shap.summary_plot(shap_values, x)
