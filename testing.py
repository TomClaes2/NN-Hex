import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import operator


moves = [-2,-1,0,1,2]
def init_board():
    res = np.zeros((13,13))
    newb = res
    for i in range(6):
        res[i,5-i] = 1
    for i in range(6):
        res[i+7,12-i] = 1
    return res

def get_inputlayer(board, pos, ori):
    inp = []
    for rij in board:
        for kolom in rij:
            inp.append(kolom)
    for i in pos:
        inp.append(i)
    for i in ori:
        inp.extend(i)
    inp =np.array(inp)
    inp = inp.reshape((1,175))
    print(inp.shape)
    return inp


ori = (3, 4)
pos = ((4, 12), (6, 6))

#print(get_inputlayer(init_board(), ori, pos))
#data = np.random.random((1000, 100))
#print(data.shape)

def isnewvalid(index):
    move = moves[index]
    if move == 0:
        return False
    return True


def isvalid(outputl):
    smaloutput = outputl.copy()
    smaloutput.sort()
    print(smaloutput)
    for i, item in enumerate(smaloutput[::-1]):
        ar = np.where(outputl==item)  # get index of array with i-largest element
        # print(ar[0][0])  index in outputlayer
        if isnewvalid(ar[0][0]) is True:
            return moves[ar[0][0]]  # move
    return 0  # if no moves available, go forward

outputl = np.array([0.2,0.4,0.6,0.7,0.1])

#valid = [1  if isvalid(moves[x]) is True else 0 for x, move in enumerate(outputl)]
#print(isvalid(outputl))
def create_model(seed):
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(128, input_dim=175, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(5, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model(6)
weights = model.layers[0].get_weights()[0]
weights1 = model.layers[1].get_weights()[0]
weights2 = model.layers[2].get_weights()[0]
biases = model.layers[0].get_weights()[1]

x = [[0,1,2],[3,4,5], [6,7,8], [9,10,11]]
x = np.array(x)

x = np.delete(x, slice(0,3), axis=0)
print(x)

#print((weights.shape), weights1.shape,weights2.shape)
#print(biases.shape,biases)