from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

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
    
    # Cần 30 ngày để dự đoán
    if len(quantities) < 30:
        padding = [0] * (30 - len(quantities))
        quantities = padding + quantities
    
    last_30_days = quantities[-30:]
    
    # Chuẩn hóa
    scaled_data = SCALER.transform(np.array(last_30_days).reshape(-1, 1))
    
    # Dự đoán 7 ngày
    predicted_7_days = []
    current_input = scaled_data.reshape(1, 30) 
    
    for _ in range(7):
        next_day_scaled = AI_MODEL.predict(current_input)
        next_day_real = SCALER.inverse_transform([[next_day_scaled[0]]])
        predicted_quantity = max(0, int(round(next_day_real[0][0])))
        
        predicted_7_days.append(predicted_quantity)
        
        # Trượt cửa sổ: Bỏ ngày đầu, nhét ngày mới đoán vào cuối
        current_input = np.append(current_input[:, 1:], [[next_day_scaled[0]]], axis=1)

    total_predicted = sum(predicted_7_days)
    
    import datetime
    base = datetime.datetime.today()
    forecast_details = []
    for i in range(7):
        date_str = (base + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d")
        forecast_details.append({"date": date_str, "predicted_quantity": predicted_7_days[i]})

    return {
        "status": "success",
        "product_id": request.product_id,
        "total_predicted_7_days": total_predicted,
        "forecast_details": forecast_details
    }