from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


def build_model(input_shape):
    model = Sequential()

    model.add(
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
