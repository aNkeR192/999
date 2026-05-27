import gradio as gr
import requests

API_URL = "http://localhost:8000/predict"
API_KEY = "aNkeR192"

def predict(text, api_key, *photos):
    if api_key != API_KEY:
        return "❌ Неверный API ключ", []
    
    photo_urls = [p for p in photos if p is not None]
    
    response = requests.post(
        API_URL,
        headers={"X-API-Key": api_key},
        json={"post_id": "1", "text": text, "photos": [{"photo_id": str(i), "url": url} for i, url in enumerate(photo_urls) if url]}
    )
    
    if response.status_code != 200:
        return f"❌ Ошибка: {response.status_code}", []
    
    data = response.json()
    predictions = data.get("predictions", [])
    
    if not predictions:
        return "🔍 Ничего не найдено", []
    
    result_text = f"### 📦 Найдено объектов: {len(predictions)}\n\n"
    for p in predictions:
        result_text += f"- **{p['object_id']}**: {p['category']} → *{p['subcategory']}* (уверенность: {p['confidence']:.2f})\n"
    
    return result_text, predictions

custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 12px;
}
h1 {
    text-align: center;
    font-size: 2em;
}
"""

with gr.Blocks(title="Страйкбол Классификатор", css=custom_css) as demo:
    gr.Markdown("""
    # 🔫 СТРАЙКБОЛ КЛАССИФИКАТОР
    
    **Распознавание оружия, снаряжения и аксессуаров из твоего объявления**
    *Загрузи фото и напиши текст — нейросеть определит, что продаёшь*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            api_key_input = gr.Textbox(label="🔑 Ключ доступа", type="password", value=API_KEY, placeholder="Введи API ключ")
            text_input = gr.Textbox(label="📝 Текст объявления", lines=3, placeholder="Пример: продаю автомат Калашникова, шлем и разгрузку")
    
    with gr.Row():
        for i in range(5):
            with gr.Column(scale=1):
                globals()[f"photo_{i}"] = gr.Image(type="filepath", label=f"📷 Фото {i+1}")
    
    submit_btn = gr.Button("🔍 РАСПОЗНАТЬ", variant="primary")
    
    with gr.Row():
        output_text = gr.Markdown(label="📊 Результат")
        output_json = gr.JSON(label="📄 Подробные данные")
    
    submit_btn.click(
        predict,
        inputs=[text_input, api_key_input] + [globals()[f"photo_{i}"] for i in range(5)],
        outputs=[output_text, output_json]
    )
    
    gr.Markdown("""
    ---
    ### 📌 Примеры для проверки
    
    | Что написать | Что определит |
    |--------------|---------------|
    | `автомат ak` | Оружие → AK |
    | `тактический жилет` | Снаряжение → Жилеты |
    | `магазин и прицел` | Аксессуары → Магазины |
    | `шлем каска` | Снаряжение → Шлемы |
    | `пулемёт m249` | Оружие → Пулемёты |
    
    ---
    **by aNkeR192 | Страйкбол | 2026**
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860)