from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os
import datetime

app = FastAPI()

AI_MODEL = None
SCALER = None

class SalesData(BaseModel):
    date: str
    quantity: int

class ForecastRequest(BaseModel):
    product_id: int
    historical_data: list[SalesData]

@app.on_event("startup")
def load_ai_brain():
    global AI_MODEL, SCALER
    if os.path.exists('ai_brain.pkl'):
        with open('ai_brain.pkl', 'rb') as f:
            data = pickle.load(f)
            AI_MODEL = data['model']
            SCALER = data['scaler']
        print("✅ Đã nạp thành công bộ não AI (Neural Network) vào bộ nhớ!")
    else:
        print("⚠️ CẢNH BÁO: Chưa tìm thấy ai_brain.pkl. Hãy chạy file train.py trước!")

@app.post("/api/predict-demand")
def predict_demand(request: ForecastRequest):
    if AI_MODEL is None:
        return {"status": "error", "message": "Hệ thống AI chưa sẵn sàng."}

    quantities = [item.quantity for item in request.historical_data]
    
    # Chỉ cần đúng 4 ngày để dự đoán
    if len(quantities) < 4:
        padding = [0] * (4 - len(quantities))
        quantities = padding + quantities
    
    # Lấy CHÍNH XÁC 4 ngày cuối cùng
    last_4_days = quantities[-4:]
    
    # Chuẩn hóa mảng 4 ngày
    scaled_data = SCALER.transform(np.array(last_4_days).reshape(-1, 1))
    
    # Đưa về ma trận 1 dòng, 4 cột để đút vào AI
    current_input = scaled_data.reshape(1, 4) 
    
    # Dự đoán 7 ngày tiếp theo
    predicted_7_days = []
    
    for _ in range(7):
        # AI đoán ngày tiếp theo (trả về số đã chuẩn hóa)
        next_day_scaled = AI_MODEL.predict(current_input)
        
        # Dịch ngược số đã chuẩn hóa về số lượng sản phẩm thật
        next_day_real = SCALER.inverse_transform([[next_day_scaled[0]]])
        predicted_quantity = max(0, int(round(next_day_real[0][0])))
        
        predicted_7_days.append(predicted_quantity)
        
        # Trượt cửa sổ: Cắt bỏ ngày cũ nhất (cột 0), nhét ngày mới đoán vào cuối
        current_input = np.append(current_input[:, 1:], [[next_day_scaled[0]]], axis=1)

    total_predicted = sum(predicted_7_days)
    
    base = datetime.datetime.today()
    forecast_details = []
    for i in range(7):
        date_str = (base + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d")
        forecast_details.append({"date": date_str, "predicted_quantity": predicted_7_days[i]})

    return {
        "status": "success",
        "product_id": request.product_id,
        "input_4_days": last_4_days, # In ra để debug xem nó lấy đúng 4 ngày chưa
        "total_predicted_7_days": total_predicted,
        "forecast_details": forecast_details
    }