from AI_model_prepare import  data_study, data_test, target_study_cat, target_test_cat, target_test
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

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


model = Sequential([
    Dense(512, activation = 'relu', input_shape = (15, )),
    Dense(256, activation = 'relu' ),
    Dense(12, activation = 'softmax')
])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights = True,
    verbose = 1
)

learn = model.fit(
    data_study, target_study_cat,
    epochs = 100,
    batch_size = 300,
    validation_split = 0.1,
    callbacks = [early_stop],
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
print(f'Conf_matrix:\n {conf_matrix}\n')

# Частые ошибки
print('\nЧастые ошибки:\n')
errors = []

for i in range(10):
    for j in range(10):
        if i != j and conf_matrix[i, j] > 0:
            errors.append((i, j, conf_matrix[i, j]))

errors.sort(key = lambda x: x[2], reverse = True)
for true, pred, errors in errors[:50]:
    print(f'{true} -> {pred}: {errors} раз')

# График conf_matrix
fig, graph = plt.subplots(figsize = (10, 8))

sns.heatmap(conf_matrix, 
            annot = True,
            fmt = 'd',
            ax = graph,
            cmap = 'RdPu',
            linewidths = 0.2,
            linecolor = 'gray',
            xticklabels = range(10),
            yticklabels = range(10))
graph.set_title('Confusion matrix', fontsize = 16, fontweight = 'bold')
graph.set_xlabel('Предсказания', fontsize = 12)
graph.set_ylabel('Верно', fontsize = 12)

plt.tight_layout()
plt.savefig('Confusion_matrix.png', dpi = 150, bbox_inches = 'tight' )

# График частых ошибок
errors_index = np.where(predict != target_test[:50])[0]
errors_conf = predictions[errors_index].max(axis = 1)
worst_errors = errors_index[np.argsort(-errors_conf)[:10]]

rows, cols = 2, 5
fig, ax = plt.subplots(figsize = (15, 7))
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.axis('off') 
for index, err_index in enumerate(worst_errors):
    row = index // cols
    col = index % cols
    x = col + 0.5
    y = rows - row - 0.5
    true_label = clothing_advice[target_test[err_index]]
    pred_label = clothing_advice[predict[err_index]]
    cloth_index = clothing_advice.index(pred_label)
    confidence = predictions[err_index][cloth_index] * 100
    rect = mpatches.FancyBboxPatch((col + 0.05, rows - row - 0.95), 0.9, 0.9, 'round, pad = 0.02', facecolor = 'pink', edgecolor = 'black', linewidth = 1)
    ax.add_patch(rect)

    ax.text(x, y, f'Верно: {true_label},\n Предсказано: {pred_label},\n  ({confidence:.0f}%)',
            ha = 'center', va = 'center', fontsize = 10)
    
    
plt.suptitle('10 уверенных ошибок', fontsize = 14, fontweight = 'bold')    
plt.tight_layout()
plt.savefig('Worst_error.png', dpi = 150, bbox_inches = 'tight' )
plt.show()

# Model 2 
model_2 = Sequential([
    Dense(512, activation = 'relu', input_shape = (15, )),
    Dropout(0.3),
    Dense(256, activation = 'relu' ),
    Dropout(0.3),
    Dense(128, activation = 'relu' ),
    Dropout(0.3),
    Dense(12, activation = 'softmax')
])
 
model_2.summary()

model_2.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

learn_2 = model_2.fit(
    data_study, target_study_cat,
    epochs = 100,
    batch_size = 300,
    validation_split = 0.1,
    callbacks = [early_stop],
    verbose = 1
) 

test_acc_2 = model_2.evaluate(data_test, target_test_cat, verbose = 0 )[1]
print(test_acc_2)
print(f'Model 2:'
    f'Test acc: {test_acc_2:.4f} ({test_acc_2 * 100:.1f}%)\n')


# Model 3
model_3 = Sequential([
    Dense(512, input_shape = (15, )),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(128, activation = 'relu' ),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(12, activation = 'softmax')
])
 
model_3.summary()

model_3.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

learn_3 = model_3.fit(
    data_study, target_study_cat,
    epochs = 100,
    batch_size = 300,
    validation_split = 0.1,
    callbacks = [early_stop],
    verbose = 1
) 

test_acc_3 = model_3.evaluate(data_test, target_test_cat, verbose = 0 )[1]
print(test_acc_3)
print(f'Model 3:'
    f'Test acc: {test_acc_3:.4f} ({test_acc_3 * 100:.1f}%)\n')