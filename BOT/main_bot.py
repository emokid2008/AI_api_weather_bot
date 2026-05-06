#init
import pickle
from keras.models import load_model
import telebot
bot = telebot.TeleBot('')
model = load_model('./AI/Ai_model.h5')
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
@bot.message_handler(commands = ['start'])
def start(message):
    bot.send_message(message.chat.id, 'Доброго времени суток, я бот - прогноз погоды,\nА также я помогаю с выбором одежды.\nЧтобы начать укажите свой город.') 
