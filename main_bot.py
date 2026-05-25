#init
import pickle
import numpy as np
import datetime
from datetime import datetime as dt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import requests
import telebot

bot = telebot.TeleBot('8568446263:AAH7aN0HUTf28Em250sBlywM_UpHORgRx6o')
model = load_model('AI/Ai_model.h5')
with open('AI/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
with open('AI/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)   
#связать ии и бота
clothing_advice = ['hot_summer',
    'shirt_shorts',
    'light_jacket_jeans',
    'raincoat_umbrella',
    'windproof_jacket',
    'light_rain_jacket',
    'medium_coat_scarf',
    'warm_winter_coat',
    'winter_wet_gear',
    'extreme_cold_gear']

advice =  [
    'Жаркое лето. Подойдёт футболка/майка и шорты. А также кепка и солнцезащитные очки.',
    'Тёплая погода. Футболка шорты/штаны',
    'Прохладная погода. Подойдёт лёгкая куртка и джинсы.',
    'На улице додждь. Подойдёт дождевик и зонт.',
    'Сильный ветер. Ветровка и джинсы.',
    'Возможен дождь. Водонепроницаемая куртка и джинсы отлично подойдут.',
    'Осень. Подойдут пальто и шарф.',
    'Холодная зима. Тёплая зимняя куртка.',
    'Cнег. Водонепроницаемая тёплая куртка и плотные штаны.',
    'Очень холодно. Супер тёплая одежда.'
]


urlParams = {
     'key': '06eaefda99bdef0b3100c8a6d5d0467a',
     'units': 'metric',
     'lang': 'ru'
 }   

@bot.message_handler(commands = ['start'])
def start(message):
    bot.send_message(message.chat.id, 'Доброго времени суток, я бот - прогноз погоды,\nА также я помогаю с выбором одежды.\nЧтобы начать укажите свой город.') 

@bot.message_handler(content_types = ['text'])
def getCity(message):

    global urlParams
    city = message.text
    url = f'https://api.openweathermap.org/data/2.5/weather?q='+city+'&units='+urlParams['units']+'&lang='+urlParams['lang']+'&appid='+urlParams['key']

    dataRes = requests.get(url)
    if dataRes.status_code == 200:
        wheatherData = dataRes.json()
        print(wheatherData)
    
        advice = weatherArr(wheatherData)

        main = wheatherData['weather'][0]['description']
        temp = round(wheatherData['main']['temp'])
        pressure = round(wheatherData['main']['pressure'])
        humid = round(wheatherData['main']['humidity'])
        temp_max = round(wheatherData['main']['temp_max'])
        temp_min = round(wheatherData['main']['temp_min'])
        wind = round(wheatherData['wind']['speed'])
        clouds = round(wheatherData['clouds']['all'])
        id = wheatherData['weather'][0]['id']

        timezone = wheatherData['timezone'] / 3600 
        if timezone > 0:
            timezoneznak = '+'
        else:
            timezoneznak = '-' 
        offset = datetime.timezone(datetime.timedelta(hours = timezone))
        sunrise = dt.fromtimestamp(wheatherData['sys']['sunrise'], offset).strftime('%H:%M')     
        sunset = dt.fromtimestamp(wheatherData['sys']['sunset'], offset).strftime('%H:%M') 

        date = dt.now(offset).strftime('%d %B %Y')
        dateTime = dt.now(offset).strftime('%H:%M') 

        if id >= 200 and id <= 211:
            imageUrl = '../картинки/дождь и гроза.jpg'
            image = open(imageUrl, 'rb')

        if id >= 212 and id <= 232:
            imageUrl = 'картинки/просто гроза.jpg'
            image = open(imageUrl, 'rb')    

        if id >= 300 and id <= 321:
            imageUrl = 'картинки/дождь.jpg'
            image = open(imageUrl, 'rb')    

        if id >= 500 and id <= 510:
            imageUrl = 'картинки/сильный дождик.jpg'
            image = open(imageUrl, 'rb')

        if id == 511:
            imageUrl = 'картинки/снег с дождем.jpg'
            image = open(imageUrl, 'rb')    

        if id >= 511 and id <= 531:
            imageUrl = 'картинки/сильный дождик.jpg'
            image = open(imageUrl, 'rb')    
 
        if id >= 600 and id <= 622:
            imageUrl = 'картинки/снег.jpg'
            image = open(imageUrl, 'rb')  
        
        if id >= 701 and id <= 741:
            imageUrl = 'картинки/туман.jpg'
            image = open(imageUrl, 'rb')      
        
        if id >= 751 and id <= 781:
            imageUrl = 'картинки/ветер.jpg'
            image = open(imageUrl, 'rb')  

        if id == 800:
            imageUrl = 'картинки/солнечно.jpg'
            image = open(imageUrl, 'rb')      

        if id == 801:
            imageUrl = 'картинки/микро облака.jpg'
            image = open(imageUrl, 'rb')      

        if id == 802:
            imageUrl = 'картинки/50проц облака.jpg'
            image = open(imageUrl, 'rb')      

        if id == 803:
            imageUrl = 'картинки/75 проц облака.jpg'
            image = open(imageUrl, 'rb')       

        if id == 804:
            imageUrl = 'картинки/облака сильные.jpg'
            image = open(imageUrl, 'rb')                     

        messageText = f'Отлично, вот погода в вашем городе:\n' \
                      f'📅Дата: {date}\n' \
                      f'⏰Время: {dateTime}\n' \
                      f'🏡Город: {city} - {main}\n' \
                      f'🌡Температура: {temp}℃\n' \
                      f'🔺Максимальная температура: {temp_max}\n' \
                      f'🔻Минимальная температура: {temp_min}\n' \
                      f'📍Давление: {pressure}\n' \
                      f'💦Влажность: {humid}\n' \
                      f'💨Ветер: {wind}км/ч\n' \
                      f'☁Облака: {clouds}\n' \
                      f'🌇Рассвет {sunrise}\n' \
                      f'🌆Закат: {sunset}'
        bot.send_photo(message.chat.id, image, caption = messageText, parse_mode = 'html')
        bot.send_message(message.chat.id, 'Совет от ИИ 🐕: ' + advice)
    else:
        bot.send_message(message.chat.id, 'Такого города нет!')
        bot.register_next_step_handler(message, getCity)

def weatherArr(wheatherData):
    main_vect = vectorizer.transform([wheatherData['weather'][0]['main']]).toarray()
    arr = np.array([wheatherData['weather'][0]['id'], wheatherData['main']['temp'], wheatherData['main']['pressure'], wheatherData['main']['humidity'],wheatherData['main']['temp_min'], wheatherData['main']['temp_max'], wheatherData['wind']['speed'], wheatherData['clouds']['all']])
    result = np.concatenate([main_vect[0], arr])
    ar = np.array([result])
    ar = scaler.transform(ar)
    print(ar)
    predictions = model.predict(ar)[0]
    predict = np.argmax(predictions)
    print(predictions)
    print(predict)
    return advice[predict]
    
    

bot.infinity_polling()                      