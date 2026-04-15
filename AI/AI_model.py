#init
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
import pickle 


data_frame = pd.read_csv('dataset.csv')
print(data_frame.keys())

data = data_frame.drop(columns = ['clothing_advice','label']).values
target = data_frame['label'].values
print(target)
vectorizer = TfidfVectorizer(max_features = 1000)

data_scaled = vectorizer.fit_transform(data).toarray()

data_study, data_test, target_study, target_test = train_test_split(data_scaled, target, test_size = 0.2, random_state = 42, stratify = target)

target_study_cat = to_categorical(target_study, num_classes = 12)
target_test_cat = to_categorical(target_test, num_classes = 12)

with open ('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)