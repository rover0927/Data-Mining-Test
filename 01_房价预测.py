import numpy as np
from sklearn.linear_model import LinearRegression

# 特征
X = np.array([
    [50,3,0.5,5],
    [60,4,0.6,6],
    [70,5,0.7,7],
    [80,6,0.8,8],
    [90,7,0.9,9],
    [100,8,1,10],
])

# 目标值
y= np.array([120,140,160,180,200,220])

model = LinearRegression()
# 训练模型
model.fit(X,y)

feature_names = ["price", "area", "rooms", "age"]
print("模型各系数：")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"模型偏置b： {model.intercept_:.2f}")

new_house = np.array([[80,6,0.8,8]])
print("预测房价：", model.predict(new_house))