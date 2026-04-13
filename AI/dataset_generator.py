import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)
# В каком порядке он записывает данные советы

# Целевое распределение: ~1000 на класс = 10_000 всего
TARGET_PER_CLASS = 1000
CLASSES = [
    'hot_summer', 'tshirt_shorts', 'light_jacket_jeans', 
    'raincoat_umbrella', 'windproof_jacket', 'light_rain_jacket',
    'medium_coat_scarf', 'warm_winter_coat', 'winter_wet_gear', 'extreme_cold_gear'
]

def generate_sample_for_class(cls):
    """Генерирует реалистичные погодные данные под конкретный класс одежды"""
    if cls == 'hot_summer':
        t = np.random.uniform(28, 40)
        hum = np.random.uniform(30, 70)
        ws = np.random.uniform(0, 8)
        main = 'Clear'
    elif cls == 'tshirt_shorts':
        t = np.random.uniform(20, 30)
        hum = np.random.uniform(40, 80)
        ws = np.random.uniform(0, 10)
        main = random.choice(['Clear', 'Clouds'])
    elif cls == 'light_jacket_jeans':
        t = np.random.uniform(10, 20)
        hum = np.random.uniform(50, 85)
        ws = np.random.uniform(2, 12)
        main = random.choice(['Clouds', 'Clear', 'Drizzle'])
    elif cls == 'raincoat_umbrella':
        t = np.random.uniform(5, 20)
        hum = np.random.uniform(75, 100)
        ws = np.random.uniform(3, 15)
        main = 'Rain'
    elif cls == 'windproof_jacket':
        t = np.random.uniform(8, 18)
        hum = np.random.uniform(40, 70)
        ws = np.random.uniform(12, 25)  # ключевой признак!
        main = 'Clouds'
    elif cls == 'light_rain_jacket':
        t = np.random.uniform(18, 28)
        hum = np.random.uniform(70, 95)
        ws = np.random.uniform(2, 10)
        main = random.choice(['Rain', 'Drizzle'])
    elif cls == 'medium_coat_scarf':
        t = np.random.uniform(0, 10)
        hum = np.random.uniform(50, 80)
        ws = np.random.uniform(1, 10)
        main = random.choice(['Clouds', 'Clear', 'Snow'])
    elif cls == 'warm_winter_coat':
        t = np.random.uniform(-10, 0)
        hum = np.random.uniform(40, 75)
        ws = np.random.uniform(1, 12)
        main = random.choice(['Snow', 'Clear'])
    elif cls == 'winter_wet_gear':
        t = np.random.uniform(-10, 0)
        hum = np.random.uniform(75, 100)  # высокая влажность
        ws = np.random.uniform(2, 15)
        main = random.choice(['Snow', 'Drizzle'])
    elif cls == 'extreme_cold_gear':
        t = np.random.uniform(-30, -10)
        hum = np.random.uniform(30, 80)
        ws = np.random.uniform(0, 20)
        main = random.choice(['Snow', 'Clear'])
    
    # Генерация остальных признаков
    pressure = np.random.uniform(980, 1040)
    clouds = np.random.uniform(0, 100) if main != 'Clear' else np.random.uniform(0, 30)
    temp_min = t - np.random.uniform(0.5, 3.0)
    temp_max = t + np.random.uniform(0.5, 3.0)
    wind_speed = ws
    base_ts = 1704067200
    sunrise = base_ts + np.random.uniform(6*3600, 9*3600)
    sunset = base_ts + np.random.uniform(16*3600, 21*3600)
    weather_id = {'Clear':800,'Clouds':800,'Rain':500,'Drizzle':300,'Snow':600}.get(main, 800) + random.randint(0,99)
    
    return {
        'main': main, 'id': weather_id, 'temp': round(t,1), 'pressure': round(pressure,1),
        'humidity': round(hum,1), 'temp_min': round(temp_min,1), 'temp_max': round(temp_max,1),
        'wind_speed': round(wind_speed,1), 'clouds': round(clouds,1),
        'sunset': int(sunset), 'sunrise': int(sunrise), 'clothing_advice': cls
    }

# Генерация сбалансированного датасета
data = []
for cls in CLASSES:
    for _ in range(TARGET_PER_CLASS):
        data.append(generate_sample_for_class(cls))

df = pd.DataFrame(data)
# Перемешиваем
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset.csv', index=False, encoding='utf-8')

print(f"✅ Сбалансированный датасет: {df.shape[0]} строк")
print(f"📊 Распределение:\n{df['clothing_advice'].value_counts()}")

