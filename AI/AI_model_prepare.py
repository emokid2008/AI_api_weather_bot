#init
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
import pickle 
import numpy as np


data_frame = pd.read_csv('dataset.csv')
print(data_frame.keys())

data = data_frame.drop(columns = ['main','clothing_advice','label']).values
target = data_frame['label'].values
print(target)
vectorizer = TfidfVectorizer(max_features = 1000)

data_scaled = vectorizer.fit_transform(data_frame['main']).toarray()
data = np.hstack((data_scaled, data))
print(data_scaled)
print(data)

data_study, data_test, target_study, target_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify = target)
print(data_study.shape, target_study.shape)

scaler = StandardScaler()
data_study = scaler.fit_transform(data_study)
data_test = scaler.transform(data_test)

target_study_cat = to_categorical(target_study, num_classes = 12)
target_test_cat = to_categorical(target_test, num_classes = 12)

with open ('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)