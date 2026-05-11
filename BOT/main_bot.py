#init
import pickle
import datetime
from datetime import datetime as dt
from keras.models import load_model
import requests
import telebot
bot = telebot.TeleBot('')
model = load_model('./AI/Ai_model.h5')
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

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

        main = wheatherData['wheather'][0]['main']
        temp = round(wheatherData['main']['temp'])
        pressure = round(wheatherData['main']['pressure'])
        humid = round(wheatherData['main']['humidity'])
        temp_max = round(wheatherData['main']['temp_max'])
        temp_min = round(wheatherData['main']['temp_min'])
        wind = round(wheatherData['wind']['speed'])
        clouds = round(wheatherData['clouds']['all'])
      #+ картинки + бейсик мессендж 

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