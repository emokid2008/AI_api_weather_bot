import pandas as pd
import numpy as np
import random
import json

np.random.seed(42)
random.seed(42)

TARGET_PER_CLASS = 1000
CLASSES = [
    'hot_summer', 'tshirt_shorts', 'light_jacket_jeans', 
    'raincoat_umbrella', 'windproof_jacket', 'light_rain_jacket',
    'medium_coat_scarf', 'warm_winter_coat', 'winter_wet_gear', 'extreme_cold_gear'
]

# 🔑 Создаём маппинг: имя совета -> числовая метка (0-9)
CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASSES)}
LABEL_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_LABEL.items()}

def generate_sample_for_class(cls):
    """Генерирует реалистичные погодные данные под конкретный класс одежды"""
    if cls == 'hot_summer':
        t, hum, ws, main = np.random.uniform(28,40), np.random.uniform(30,70), np.random.uniform(0,8), 'Clear'
    elif cls == 'tshirt_shorts':
        t, hum, ws, main = np.random.uniform(20,30), np.random.uniform(40,80), np.random.uniform(0,10), random.choice(['Clear','Clouds'])
    elif cls == 'light_jacket_jeans':
        t, hum, ws, main = np.random.uniform(10,20), np.random.uniform(50,85), np.random.uniform(2,12), random.choice(['Clouds','Clear','Drizzle'])
    elif cls == 'raincoat_umbrella':
        t, hum, ws, main = np.random.uniform(5,20), np.random.uniform(75,100), np.random.uniform(3,15), 'Rain'
    elif cls == 'windproof_jacket':
        t, hum, ws, main = np.random.uniform(8,18), np.random.uniform(40,70), np.random.uniform(12,25), 'Clouds'
    elif cls == 'light_rain_jacket':
        t, hum, ws, main = np.random.uniform(18,28), np.random.uniform(70,95), np.random.uniform(2,10), random.choice(['Rain','Drizzle'])
    elif cls == 'medium_coat_scarf':
        t, hum, ws, main = np.random.uniform(0,10), np.random.uniform(50,80), np.random.uniform(1,10), random.choice(['Clouds','Clear','Snow'])
    elif cls == 'warm_winter_coat':
        t, hum, ws, main = np.random.uniform(-10,0), np.random.uniform(40,75), np.random.uniform(1,12), random.choice(['Snow','Clear'])
    elif cls == 'winter_wet_gear':
        t, hum, ws, main = np.random.uniform(-10,0), np.random.uniform(75,100), np.random.uniform(2,15), random.choice(['Snow','Drizzle'])
    elif cls == 'extreme_cold_gear':
        t, hum, ws, main = np.random.uniform(-30,-10), np.random.uniform(30,80), np.random.uniform(0,20), random.choice(['Snow','Clear'])
    else:
        raise ValueError(f"Неизвестный класс: {cls}")
    
    pressure = np.random.uniform(980, 1040)
    clouds = np.random.uniform(0, 100) if main != 'Clear' else np.random.uniform(0, 30)
    temp_min = t - np.random.uniform(0.5, 3.0)
    temp_max = t + np.random.uniform(0.5, 3.0)
    base_ts = 1704067200
    sunrise = base_ts + np.random.uniform(6*3600, 9*3600)
    sunset = base_ts + np.random.uniform(16*3600, 21*3600)
    weather_id = {'Clear':800,'Clouds':800,'Rain':500,'Drizzle':300,'Snow':600}.get(main, 800) + random.randint(0,99)
    
    return {
        'main': main, 'id': weather_id, 'temp': round(t,1), 'pressure': round(pressure,1),
        'humidity': round(hum,1), 'temp_min': round(temp_min,1), 'temp_max': round(temp_max,1),
        'wind_speed': round(ws,1), 'clouds': round(clouds,1),
        'sunset': int(sunset), 'sunrise': int(sunrise), 'clothing_advice': cls
    }

# 1. Генерация
data = []
for cls in CLASSES:
    for _ in range(TARGET_PER_CLASS):
        data.append(generate_sample_for_class(cls))

df = pd.DataFrame(data)

# 2. Добавление числовой метки (0-9) строго по порядку CLASSES
df['label'] = df['clothing_advice'].map(CLASS_TO_LABEL)

# 3. Перемешивание и сохранение
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset.csv', index=False, encoding='utf-8')

# 4. Сохранение маппинга для Telegram-бота (критично!)
mapping = {"class_to_label": CLASS_TO_LABEL, "label_to_class": {str(k):v for k,v in LABEL_TO_CLASS.items()}}
with open('label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)

print(f"✅ Датасет сохранён: {df.shape[0]} строк")
print(f"🔢 Маппинг меток (0→9):\n{CLASS_TO_LABEL}")
print(f"📊 Проверка баланса:\n{df['label'].value_counts().sort_index()}")