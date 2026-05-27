import torch
from PIL import Image
from torchvision import transforms
import httpx
import io
from taxonomy import text_to_subcategory

IMG_SIZE = 224
CATEGORY_NAMES = ["Страйкбольное оружие", "Снаряжение и защита", "Аксессуары и Запчасти"]

SUBCATEGORY_MAP = {
    0: "AK",
    1: "HK",
    2: "M-series",
    3: "helmet",
    4: "vest",
    5: "pouch",
    6: "backpack",
    7: "machinegun",
    8: "pistol",
    9: "rifle",
    10: "shotgun",
}

SUBCAT_TO_CATEGORY = {
    "AK": "Страйкбольное оружие",
    "HK": "Страйкбольное оружие",
    "M-series": "Страйкбольное оружие",
    "machinegun": "Страйкбольное оружие",
    "pistol": "Страйкбольное оружие",
    "rifle": "Страйкбольное оружие",
    "shotgun": "Страйкбольное оружие",
    "helmet": "Снаряжение и защита",
    "vest": "Снаряжение и защита",
    "pouch": "Снаряжение и защита",
    "backpack": "Снаряжение и защита",
}

RUSSIAN_KEYWORDS = {
    "Страйкбольное оружие": ["ak", "hk", "m4", "m16", "винтовка", "автомат", "оружие", "калаш", "калашников", "ар15", "cyma", "vfc", "дробовик", "пистолет", "снайпер", "пулемёт"],
    "Снаряжение и защита": ["жилет", "разгрузка", "вест", "шлем", "каска", "бронежилет", "наколенник", "налокотник", "перчатка", "тактический", "подсумок", "рюкзак"],
    "Аксессуары и Запчасти": ["магазин", "прицел", "оптика", "фонарь", "лцу", "лазер", "аккумулятор", "ремни", "антабка", "глушитель", "ствол"]
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def translate_category_from_russian(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in RUSSIAN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    return "Аксессуары и Запчасти"

def load_image_from_url(url: str):
    with httpx.Client() as client:
        response = client.get(url, timeout=10.0)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img

def predict_subcategory(model, image, device="cpu"):
    if isinstance(image, str):
        if image.startswith("http"):
            image = load_image_from_url(image)
        else:
            image = Image.open(image).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, dim=1).item()
    return SUBCATEGORY_MAP[pred], torch.softmax(outputs, dim=1)[0][pred].item()

def predict_categories(model, tokenizer, text, device="cpu", threshold=0.5):
    if model is None or tokenizer is None:
        return [translate_category_from_russian(text)]
    
    try:
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = outputs.cpu().numpy()[0]

        result = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                result.append(CATEGORY_NAMES[i])
        
        if not result:
            result = [translate_category_from_russian(text)]
        
        return result
    except:
        return [translate_category_from_russian(text)]

def predict_post(category_model, subcategory_model, tokenizer, text, photo_urls, device="cpu", threshold=0.5):
    predicted_categories = predict_categories(category_model, tokenizer, text, device, threshold)

    predictions = []
    for i, url in enumerate(photo_urls):
        try:
            subcat, conf = predict_subcategory(subcategory_model, url, device)
            category = SUBCAT_TO_CATEGORY.get(subcat, predicted_categories[0] if predicted_categories else "Аксессуары и Запчасти")

            predictions.append({
                "object_id": str(i + 1),
                "category": category,
                "subcategory": subcat,
                "confidence": round(conf, 3),
                "photo_ids": [str(i + 1)]
            })
        except Exception as e:
            continue

    if not predictions and predicted_categories:
        for cat in predicted_categories:
            subcat = text_to_subcategory(text, cat)
            predictions.append({
                "object_id": "text_only",
                "category": cat,
                "subcategory": subcat,
                "confidence": 0.6,
                "photo_ids": []
            })

    return predictions