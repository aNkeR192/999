from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import gradio as gr
import threading
import uvicorn

app = FastAPI()
API_KEY = "aNkeR192"

KEYWORDS = {
    "Страйкбольное оружие": ["ak", "hk", "m4", "винтовка", "автомат", "пистолет", "дробовик", "калаш"],
    "Снаряжение и защита": ["жилет", "шлем", "каска", "разгрузка", "бронежилет", "подсумок", "рюкзак"],
    "Аксессуары и Запчасти": ["магазин", "прицел", "оптика", "фонарь", "лазер", "аккумулятор", "ремни"]
}

SUBCAT = {
    "ak": "AK", "hk": "HK", "m4": "M4", "винтовка": "Винтовка",
    "жилет": "Жилет", "шлем": "Шлем", "разгрузка": "Разгрузка", "рюкзак": "Рюкзак",
    "магазин": "Магазин", "прицел": "Прицел", "фонарь": "Фонарь"
}

def predict_category(text: str) -> str:
    text_lower = text.lower()
    for cat, words in KEYWORDS.items():
        for word in words:
            if word in text_lower:
                return cat
    return "Аксессуары и Запчасти"

def predict_subcategory(text: str) -> str:
    text_lower = text.lower()
    for key, subcat in SUBCAT.items():
        if key in text_lower:
            return subcat
    return "По тексту"

class Photo(BaseModel):
    photo_id: str
    url: str

class PredictRequest(BaseModel):
    post_id: str
    text: str
    photos: List[Photo] = []

@app.post("/predict")
async def predict(request: PredictRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    
    category = predict_category(request.text)
    subcategory = predict_subcategory(request.text)
    
    predictions = [{
        "object_id": "1",
        "category": category,
        "subcategory": subcategory,
        "confidence": 0.8,
        "photo_ids": [p.photo_id for p in request.photos]
    }]
    
    return {"post_id": request.post_id, "predictions": predictions}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Gradio интерфейс
def gradio_predict(text, api_key):
    if api_key != API_KEY:
        return "❌ Неверный ключ"
    category = predict_category(text)
    subcategory = predict_subcategory(text)
    return f"**Категория:** {category}\n**Подкатегория:** {subcategory}\n**Уверенность:** 0.8"

with gr.Blocks(title="Страйкбол Классификатор", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔫 Страйкбол Классификатор\n*by aNkeR192*")
    gr.Markdown("Введите текст объявления и получите категорию и подкатегорию снаряжения.")
    
    with gr.Row():
        api_input = gr.Textbox(label="🔑 API Key", type="password", value=API_KEY)
        text_input = gr.Textbox(label="📝 Текст объявления", lines=3, placeholder="Пример: продаю автомат AK и тактический жилет")
    
    output = gr.Markdown(label="📊 Результат")
    btn = gr.Button("🚀 Распознать", variant="primary")
    btn.click(gradio_predict, inputs=[text_input, api_input], outputs=output)
    
    gr.Markdown("---\n**Примеры:** `автомат ak` → Оружие → AK | `тактический жилет` → Снаряжение → Жилет")

# Запуск
if __name__ == "__main__":
    threading.Thread(target=lambda: demo.launch(server_name="0.0.0.0", server_port=7860)).start()
    uvicorn.run(app, host="0.0.0.0", port=10000)
