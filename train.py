import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import os

print("🔄 BƯỚC 1: Đang chuẩn bị dữ liệu học...")
np.random.seed(42)
days = np.arange(365)
# Dữ liệu có tính chu kỳ + xu hướng tăng + nhiễu
sales = 50 + 20 * np.sin(days / 7) + days * 0.1 + np.random.normal(0, 5, 365)
sales = np.maximum(sales, 0)

data = pd.DataFrame({'sales': sales})

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Tạo cửa sổ trượt (30 ngày trước đoán ngày tiếp theo)
X_train = []
y_train = []
time_step = 30

for i in range(time_step, len(scaled_data)):
    X_train.append(scaled_data[i-time_step:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print("🧠 BƯỚC 2: Đang khởi tạo Mạng Nơ-ron Nhân tạo (MLP)...")
# Khởi tạo Mạng Nơ-ron với 2 lớp ẩn (Mỗi lớp 50 nơ-ron)
# Bật verbose=True để in ra quá trình học cho thầy xem
model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=200, random_state=42, verbose=True)

print("🔥 BƯỚC 3: Bắt đầu quá trình huấn luyện (Training)...")
model.fit(X_train, y_train)

print("✅ Hoàn tất! Đang lưu bộ não AI...")
# Lưu bộ não và cả cái công cụ chuẩn hóa (scaler) để dùng cho web
with open('ai_brain.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

# Vẽ biểu đồ Loss
plt.plot(model.loss_curve_, color='red', label='Sai số (Loss)')
plt.title('Biểu đồ quá trình học tập của AI (Neural Network)')
plt.xlabel('Vòng lặp (Iterations)')
plt.ylabel('Mức độ sai lệch')
plt.legend()
plt.savefig('training_proof.png')
print("📸 Đã lưu ảnh biểu đồ chứng minh vào 'training_proof.png'")