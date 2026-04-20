from AI_model_prepare import  data_study, data_test, target_study_cat, target_test_cat, target_test
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

#predict 
predictions = model.predict(data_test[:50], verbose = 0)
predict = np.argmax(predictions, axis = 1)

for i in range(50):
    pred = np.argmax(predictions[i])
    true = target_test[i]
    conf = predictions[i][pred]

    result = 'Верно' if pred == true else 'Неверно'

    print(f'{result}:\n'
          f'Предсказано: {pred}\n'
          f'Верно: {true}\n'
          f'Уверенность: {conf*100:.1f}\n')
    
# График обучения 
fig, (acc, loss) = plt.subplots(1, 2, figsize = (14, 5))

# Acc
acc.plot(learn.history['accuracy'], label = 'Точность обучение', linewidth = 2)
acc.plot(learn.history['val_accuracy'], label = 'Точность после обучения', linewidth = 2)
acc.set_title('Точность модели', fontsize = 14, fontweight = 'bold')
acc.set_xlabel('Эпохи')
acc.set_ylabel('Точность')
acc.legend()
acc.grid(True, alpha = 0.3)


# loss
loss.plot(learn.history['loss'], label = 'Потери изучения', linewidth = 2)
loss.plot(learn.history['val_loss'], label = 'Потери после изучения', linewidth = 2)
loss.set_title('Потери модели', fontsize = 14, fontweight = 'bold')
loss.set_xlabel('Эпохи')
loss.set_ylabel('Потери')
loss.legend()
loss.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('model_fit.png', dpi = 150, bbox_inches = 'tight' )

# conf_matrix
conf_matrix = confusion_matrix(target_test[:50], predict)
print(f'Conf_matrix:\n {conf_matrix}')

# Частые ошибки
print(f'Частые ошибки:')
errors = []

for i in range(10):
    for j in range(10):
        if i != j and conf_matrix[i, j] > 0:
            errors.append((i, j, conf_matrix[i, j]))

errors.sort(key = lambda x: x[2], reverse = True)
for true, pred, errors in errors[:50]:
    print(f'{true} -> {pred}: {errors} раз')

