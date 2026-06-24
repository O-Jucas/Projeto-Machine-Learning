import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def encode_labels(labels, image_array, words_to_keep):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)

    # Adiciona dimensão de canal para CNN: (N, 64, 64) -> (N, 64, 64, 1)
    X = np.stack(image_array)[..., np.newaxis]
    y = encoded

    X_train, X_validate, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = len(words_to_keep)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_validate = to_categorical(y_test, num_classes=num_classes)

    print(y_train[-1])

    return X_train, X_validate, y_train, y_validate


def create_cnn_model_1(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_cnn_model_2(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def create_cnn_model_3(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def training(model, X_train, y_train, X_validate, y_validate):

    early_stopping = EarlyStopping(
        min_delta=0.005,
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validate, y_validate),
        epochs=15,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Retornando o histórico de loss e accuracy para os gráficos
    return history.history


def analyze_training(history, caminho_output, modelo_num):
    history_frame = pd.DataFrame(history)

    fig, ax = plt.subplots()
    history_frame.loc[:, ['loss', 'val_loss']].plot(ax=ax)
    ax.set_title(f"Modelo {modelo_num} - Loss")
    fig.savefig(caminho_output / f"modelo{modelo_num}_loss.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot(ax=ax)
    ax.set_title(f"Modelo {modelo_num} - Accuracy")
    fig.savefig(caminho_output / f"modelo{modelo_num}_accuracy.png")
    plt.close(fig)