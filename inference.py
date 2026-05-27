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

def predict_categories_from_text(text: str) -> list:
    """Парсит текст и возвращает список категорий (мультилейбл)"""
    text_lower = text.lower()
    categories = []
    
    if any(word in text_lower for word in ["ak", "hk", "m4", "винтовка", "автомат", "калаш", "оружие", "дробовик", "пистолет", "пулемёт"]):
        categories.append("Страйкбольное оружие")
    
    if any(word in text_lower for word in ["жилет", "разгрузка", "вест", "шлем", "каска", "бронежилет", "подсумок", "рюкзак"]):
        categories.append("Снаряжение и защита")
    
    if any(word in text_lower for word in ["магазин", "прицел", "оптика", "фонарь", "лазер", "аккумулятор", "ремни"]):
        categories.append("Аксессуары и Запчасти")
    
    return categories if categories else ["Аксессуары и Запчасти"]

def predict_categories(model, tokenizer, text, device="cpu", threshold=0.5):
    # Используем keyword-парсинг для мультилейбл
    return predict_categories_from_text(text)

def predict_post(category_model, subcategory_model, tokenizer, text, photo_urls, device="cpu", threshold=0.5):
    predicted_categories = predict_categories(category_model, tokenizer, text, device, threshold)

    predictions = []
    object_id = 1
    
    for cat in predicted_categories:
        subcat = text_to_subcategory(text, cat)
        predictions.append({
            "object_id": str(object_id),
            "category": cat,
            "subcategory": subcat,
            "confidence": 0.8,
            "photo_ids": []
        })
        object_id += 1

    return predictions
