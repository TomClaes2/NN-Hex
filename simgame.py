import simn
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import operator


def generate_start():
    arr = np.empty((0,261))
    zeros = np.zeros((47, 128))
    zeros2 = np.zeros((47, 5))
    for i in range(128):
        array = arr
        # create random nn and save in array
        # per model 3 matrices met weights; (175,128), (128,128), (128,5)
        model = load_models()

        w = model.layers[0].get_weights()[0]
        w1 = model.layers[1].get_weights()[0]
        w2 = model.layers[2].get_weights()[0]

        #model.save_weights("models/model%s.h5" % (str(i)))
        arr = w_save_array(w, w1, w2, zeros, zeros2, array)  # weights worden doorgegeven samen met zero arrays

    print(arr.shape)
    print(save_array(arr))

    return None  # dimensions (n,3) n rijen want n nets aangemaakt en 3 kolommen voor de 3 weight matrices


def create_model(seed):
    seed = 6
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(128, input_dim=175, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(5, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.layers[0].get_weights()[1].shape, 'weights')  # weights
    #w1 = np.random.uniform(0,1,(175,128))

    #model.layers[0].set_weights()[0]
    #model.layers[0].set_weights()[0]
    #
    # model.layers[0].set_weights()[0]




    #model_json = model.to_json()
    #with open("models/model.json", "w") as json_file:
     #   json_file.write(model_json)
    return model



def save_models():
    modlist = generate_start()
    np.save("modelstartlist.npy", modlist)
    return "Saved!"


def load_models():
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model


def w_save_array(w0, w1, w2, zeros, zeros2, array):  # w1 en 22 hebben 48 rijen minder
    w1 = np.append(w1, zeros, axis=0)
    w2 = np.append(w2, zeros2, axis=0)

    wtot = np.append(w0, w1, axis=1)
    wtot = np.append(wtot, w2, axis=1)

    array = np.append(array, wtot, axis=0)
    return array


def setmodel(model, modlist):
    for x in range(0, modlist.shape[1], 3):  # x is telkens index van eerste layer; opslaan in numpy array .npy
        for i in range(3):
            pass


def save_array(array):
    np.save('Nets.npy', array)
    return 'Saved!'


def load_array():
    ar = np.load('Nets.npy')
    return ar


def first_array():
    model = create_model(5)

    w = model.layers[0].get_weights()[0]
    w1 = model.layers[1].get_weights()[0]
    w2 = model.layers[2].get_weights()[0]

    zeros = np.zeros((47, 128))
    zeros2 = np.zeros((47, 5))

    # model.save_weights("models/model%s.h5" % (str(i)))
    array = np.empty((0,261))

    arr = w_save_array(w, w1, w2, zeros, zeros2, array)  # weights worden doorgegeven samen met zero arrays

    print(save_array(arr))
print(generate_start())
print(load_array().shape)

#print(save_models())
#print(generate_start())
#print(load_models("modelstartlist.npy"))

'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print('loaded')
print(loaded_model.layers[0].get_weights()[0].shape)
'''

print(create_model(5))
'''
lst = generate_start()
lst = np.array(lst)
print(lst)
print(lst.shape)
print(lst[0,0].shape)  # eerste item in array, matrix van 175x128
for x in range(0, lst.shape[1], 3):  # x is telkens index van eerste layer; opslaan in numpy array .npy
    print(x, lst[0,x].shape)
    print(x+1, lst[0,x+1].shape)
    print(x+2, lst[0,x+2].shape)
'''