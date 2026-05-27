CATEGORIES = {
    0: "Страйкбольное оружие",
    1: "Снаряжение и защита",
    2: "Аксессуары и Запчасти",
}

CATEGORY_NAME_TO_ID = {v: k for k, v in CATEGORIES.items()}

SUBCATEGORY_FOLDER_MAP = {
    "ak": "AK", "HK": "HK", "M serias": "M-series",
    "helmet": "Шлемы", "vest": "Жилеты", "pouch": "Подсумки",
    "backpack": "Рюкзаки", "machinegun": "Пулемёты",
    "pistol": "Пистолеты", "rifle": "Винтовки", "shotgun": "Дробовики",
}

GEAR_SUBCATEGORIES = {
    "Жилеты": ["жилет", "разгрузка", "вест", "plate carrier", "молле", "molle", "vest"],
    "Шлемы": ["шлем", "каска", "helm", "airframe", "fast", "ops-core", "helmet"],
    "Подсумки": ["подсумок", "подсумки", "pouch", "кобура"],
    "Рюкзаки": ["рюкзак", "сумка", "backpack", "bag", "camelback"],
}

ACCESSORY_SUBCATEGORIES = {
    "Магазины": ["магазин", "магазины", "magazine", "mag"],
    "Прицелы": ["прицел", "оптика", "коллиматор", "eotech", "aimpoint", "scope"],
    "Фонари": ["фонарь", "laser", "лцу", "лазер"],
    "Аккумуляторы": ["аккумулятор", "акб", "lipo", "батарея"],
}

def text_to_subcategory(text: str, category: str) -> str:
    text_lower = text.lower()
    if category == "Страйкбольное оружие":
        scores = {
            "AK": sum(1 for kw in ["ak", "ак", "акм", "rpk"] if kw in text_lower),
            "HK": sum(1 for kw in ["hk", "hk416", "g36", "mp5"] if kw in text_lower),
            "M-series": sum(1 for kw in ["m4", "m16", "ar15"] if kw in text_lower),
            "Пистолеты": sum(1 for kw in ["пистолет", "glock", "p226"] if kw in text_lower),
            "Пулемёты": sum(1 for kw in ["пулемёт", "lmg", "pkm"] if kw in text_lower),
            "Винтовки": sum(1 for kw in ["винтовка", "rifle", "карабин"] if kw in text_lower),
            "Дробовики": sum(1 for kw in ["дробовик", "shotgun"] if kw in text_lower),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Прочее оружие"
    elif category == "Снаряжение и защита":
        scores = {k: sum(1 for kw in kws if kw in text_lower) for k, kws in GEAR_SUBCATEGORIES.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Прочее снаряжение"
    elif category == "Аксессуары и Запчасти":
        scores = {k: sum(1 for kw in kws if kw in text_lower) for k, kws in ACCESSORY_SUBCATEGORIES.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Прочие аксессуары"
    return "Неизвестно"
