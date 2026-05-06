#init
import pickle
from keras.models import load_model
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
    url = 'https://api.openweathermap.org/data/2.5/weather?q={city name}&appid={API key}'

    