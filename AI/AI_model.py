from AI_model_prepare import  data_study, data_test, target_study_cat, target_test_cat
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

model = Sequential([
    Dense(256, activation = 'relu', input_shape = (15, )),
    Dense(128, activation = 'relu'),
    Dense(12, activation = 'softmax')
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

learn = model.fit(
    data_study, target_study_cat,
    epochs = 100,
    batch_size = 20,
    validation_split = 0.1,
    verbose = 1
)

test_loss, test_acc = model.evaluate(data_test, target_test_cat, verbose = 0 )
print(f'Test loss: {test_loss:.4f}\n'
      f'Test acc: {test_acc:.4f} ({test_acc * 100:.1f}%)\n')

