import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import operator


moves = [-2, -1, 0, 1, 2]
changel = [(-1,0), (-1,1), (0,1),(1,0),(1,-1), (0,-1)]  # indexen komen overeen met comp
comp = ['NW', 'NE', 'E', 'SE', 'SW', 'W']  #
def create_model():
    seed = 6
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(128, input_dim=175, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(5, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model(file, fileweight):  # "model.json"  "model.h5"
    print('start loading ...')
    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights(fileweight)
    print("Loaded model from disk")
    return loaded_model

#  print(load_model("model.json","model.h5"))

def init_board(board):
    res = np.zeros((13,13))
    newb = board[:,:,0]+board[:,:,1]
    for i in range(6):
        res[i,5-i] = 1
    for i in range(6):
        res[i+7,12-i] = 1
    return res+newb


def get_inputlayer(board, pos, ori):
    inp = []
    for rij in board:
        for kolom in rij:
            inp.append(kolom)
    for i in pos:
        inp.append(i)
    for i in ori:
        inp.extend(i)
    inp = np.array(inp)
    #print(inp.shape)
    inp = inp.reshape((1, 175))

    return inp

def check_pos(board, pos):  # return bool True/False based on pos
    # Correct !!
    if pos[0] < 0 or pos[1] < 0:  # geen negatieve indices
        #print('NN negative indice')
        return False
    try:
        _ = board[pos]  # indices zijn element board en waarden bestaan
        if board[pos] == 1:
            #print('board pos is 1')
            return False
        return True

    except:
        #print('geen board pos gevonden')
        return False

def predict_next_pos(pos, ori, move):  # slang heeft pos en ori, volgende beweging is naar vaste plaats afh van move
    # move bepalen op basis van volgende plaats
    # ori 0-->5 moet naar diff gaan, om newpos te berekenen
    # correct!!!
    newori = (move + ori) % 6
    newpos = tuple(map(operator.add, pos, changel[newori]))
    #print("newpos predicted", newpos)
    return newpos

def isvalid_largest(board, posi, ori, outputl):
    smaloutput = outputl[0].copy()
    smaloutput.sort()
    for i, item in enumerate(smaloutput[::-1]):
        #print('smalout', smaloutput[::-1])
        ar = []
        ar = np.where(outputl[0]==item)  # get index of array with i-largest element
        # print(ar[0][0])  index in outputlayer
        #print('ar', ar )
        move = moves[ar[0][0]]
        if check_pos(board, predict_next_pos(posi, ori, move)) is True:
            #print('Neural move: ',move)
            return move# move
        else:
            pass
    #print('Neural11 no options', i)
    return 0  # if no moves available, go forward


loaded_model = load_model("model.json","model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

def generate_move(board, positions, orientations):

    """Generate a move.

    Args:
        board (np.array): The playing field, represented as 3D array.
        positions (list): A list of my position, as tuple `(x, y)`, and my
            opponents position.
        orientations (list): A list of my orientation (integer in [0,5]),
            and my opponent's orientation.

    Returns:
        move (int): An integer in [-2,2].
    """

    working = init_board(board)
    inputlayer = get_inputlayer(working, orientations, positions)
    outputlayer = loaded_model.predict(inputlayer)
    ownposition = (positions[0][1],positions[0][0])
    ownorientation = orientations[0]
    #print(outputlayer, 'outl')
    #print('Neural position start', ownposition)

    move = isvalid_largest(working, ownposition, ownorientation, outputlayer)


    #print('neural move', move)
    return move
'''
ori = (3, 4)
pos = ((4, 12), (6, 6))
inputlayer = get_inputlayer(init_board(), ori, pos)
prediction = loaded_model.predict(inputlayer)
print(prediction)
testing link
'''

#C:\Users\Tom\Documents\Pythondownl\python.exe simulator.py NewNeural.py NeuralAgent.py

