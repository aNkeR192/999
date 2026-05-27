import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import torch
import gradio as gr

from model_loader import load_category_model, load_subcategory_model, load_tokenizer
from inference import predict_post

app = FastAPI(title="Airsoft Equipment Classification API")

API_KEY = os.environ.get("API_KEY", "aNkeR192")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

category_model = load_category_model(device=device)
subcategory_model = load_subcategory_model(device=device)
tokenizer = load_tokenizer()


class Photo(BaseModel):
    photo_id: str
    url: str


class PredictRequest(BaseModel):
    post_id: str
    text: str
    photos: List[Photo] = []


class PredictionItem(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]


class PredictResponse(BaseModel):
    post_id: str
    predictions: List[PredictionItem]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    photo_urls = [photo.url for photo in request.photos]

    predictions = predict_post(
        category_model=category_model,
        subcategory_model=subcategory_model,
        tokenizer=tokenizer,
        text=request.text,
        photo_urls=photo_urls,
        device=device,
        threshold=0.5
    )

    return PredictResponse(post_id=request.post_id, predictions=predictions)


@app.get("/health")
async def health():
    return {"status": "ok"}


def gradio_predict(text, api_key, *photos):
    if api_key != API_KEY:
        return "❌ Invalid API key", []

    photo_urls = [p for p in photos if p is not None]
    predictions = predict_post(
        category_model, subcategory_model, tokenizer, text, photo_urls, device, 0.5
    )

    result_text = f"🔍 Найдено объектов: {len(predictions)}\n\n"
    for p in predictions:
        result_text += f"📦 {p['object_id']}: {p['category']} → {p['subcategory']} (conf: {p['confidence']:.2f})\n"

    return result_text, predictions


with gr.Blocks(title="Airsoft Classifier") as demo:
    gr.Markdown("# Страйкбольный классификатор")
    gr.Markdown("Введите текст объявления и загрузите фото")

    with gr.Row():
        api_key_input = gr.Textbox(label="API Key", type="password", value=API_KEY)
        text_input = gr.Textbox(label="Текст объявления", lines=3)

    photo_inputs = [gr.Image(type="filepath", label=f"Фото {i + 1}") for i in range(5)]
    output_text = gr.Textbox(label="Результат", lines=10)
    output_json = gr.JSON(label="Predictions")

    submit_btn = gr.Button("Определить")
    submit_btn.click(
        gradio_predict,
        inputs=[text_input, api_key_input] + photo_inputs,
        outputs=[output_text, output_json]
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)