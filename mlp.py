import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

def train_model(training_data, testing_data, epoche):
    model = Sequential()
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    model.fit\
        (training_data, testing_data, nb_epoch=epoche, verbose=2)

    model.save('my_model.h5')

    return model

# # the four different states of the XOR gate
# training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
#
# # the four expected results in the same order
# target_data = np.array([[0],[1],[1],[0]], "float32")



# print(model.predict(training_data).round())