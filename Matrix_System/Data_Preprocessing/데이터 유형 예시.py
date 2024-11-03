# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:08:52 2024

@author: Whan Lee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 4x6 크기의 랜덤 픽셀 이미지 생성
random_image = np.random.rand(3, 4)

# 이미지 시각화 (흑백)
plt.figure(figsize=(4, 4))
plt.imshow(random_image, cmap='gray', interpolation='nearest')
plt.axis('off')  # 축 제거
plt.show()

# 픽셀 값을 데이터프레임으로 변환하여 소수 둘째 자리로 반올림
pixel_values = pd.DataFrame(random_image.round(2))

# 픽셀 값 출력
print("3x4 이미지의 각 픽셀 값:")
print(pixel_values)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

# Generate a sample dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.2, random_state=42)

# Define the model
model = GradientBoostingRegressor()

# Calculate learning curve
train_sizes, train_scores, validation_scores = learning_curve(
    model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5, scoring='neg_mean_squared_error'
)

# Calculate mean and std for train and validation scores
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = -np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='red', label='Training Score')
plt.plot(train_sizes, validation_scores_mean, 'o-', color='green', label='Validation Score')
plt.fill_between(train_sizes, 
                 validation_scores_mean - validation_scores_std, 
                 validation_scores_mean + validation_scores_std, 
                 alpha=0.2, color='gray')

# Add labels, title, and legend
plt.title('Gradient Boosting Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Negative Mean Squared Error')
plt.legend(loc='best')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:33:45 2024

@author: Whan Lee
"""
# 시계열 그래프

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(time, amplitude_time_adjusted, color='blue')
ax.set_title("Time Series Analysis with Multiple Anomalies")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

plt.show()


fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(freq[:len(freq)//2], amplitude_freq[:len(freq)//2], color='blue')
ax.set_title("Frequency Analysis")
ax.set_xlabel("Frequency")
ax.set_ylabel("Amplitude")

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:14:49 2024

@author: Whan Lee
"""
import numpy as np
import matplotlib.pyplot as plt

# 푸리에 변환 주파수 범위 설정
f = np.linspace(-5, 5, 500)
# sinc 함수 (델타 함수의 푸리에 변환)
sinc_func = np.sinc(f)

# 그래프 그리기
plt.figure(figsize=(4, 4))
plt.plot(f, sinc_func, color='blue', linewidth=1.5)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


#PCA

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the components and loadings
variables = ["X100m", "X110m_hurdle", "X400m", "X1500m", "Pole_vault", "Javeline", "Shot_put", "Long_jump", "Discus", "High_jump"]
loadings = np.array([
    [0.9, 0.1], [0.8, 0.3], [0.7, 0.2], [0.1, 0.9], [0.2, 0.8],
    [0.4, 0.7], [0.3, 0.5], [0.6, 0.4], [0.5, 0.3], [0.4, 0.6]
])
cos2 = np.square(loadings).sum(axis=1)

# Color mapping based on cos2 values
colors = plt.cm.plasma(cos2)

# Create a PCA biplot
fig, ax = plt.subplots(figsize=(8, 8))
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
ax.add_patch(circle)

# Plot vectors
for i, var in enumerate(variables):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color=colors[i], alpha=0.8, head_width=0.05)
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, var, color=colors[i], ha='center', va='center', fontsize=10)

# Axis labels and limits
plt.xlabel("Dim1 (41.2%)")
plt.ylabel("Dim2 (18.4%)")
plt.title("Variables - PCA")

# Set limits and aspect
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(True)
ax.set_aspect('equal')

# Color bar
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ticks=[0, 0.5, 0.7])
cbar.set_label('cos2')

plt.show()

# Plotting PCA projection without principal component vectors and with all points in blue

# Plotting PCA process (simplified as requested)
plt.figure(figsize=(8, 8))

# Original data points
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, color="blue", label="Original Data")

# Projected data points
plt.scatter(X_projected[:, 0], X_projected[:, 1], color="blue", alpha=0.6)

# Lines connecting original and projected points
for i in range(X.shape[0]):
    plt.plot([X[i, 0], X_projected[i, 0]], [X[i, 1], X_projected[i, 1]], "gray", alpha=0.3, linestyle="--")

# Formatting
plt.axhline(0, color='k',linewidth=0.5)
plt.axvline(0, color='k',linewidth=0.5)
plt.gca().set_aspect('equal', 'box')
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.show()
