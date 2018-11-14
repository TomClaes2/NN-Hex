import numpy as np
import operator


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import operator

ories = [0,1,2,3,4,5]
compass = ['NW', 'NE', 'E', 'SE', 'SW', 'W']
moves = [-2,-1,0,1,2]
diff = [(-1, 0),  # NW
             (-1, 1),  # NE
             (0, 1),   # E
             (1, 0),   # SE
             (1, -1),  # SW
             (0, -1)]  # W
working = None
count = 0

def init_board(board):
    res = np.zeros((13,13))
    newb = board[:,:,0]+board[:,:,1]
    for i in range(6):
        res[i,5-i] = 1
    for i in range(6):
        res[i+7,12-i] = 1
    return res+newb

def generate_move(board, positions, orientations):
    '''array vn de vorm y,x
    playing field: (0,6)-->(0,12)
    (1,5)-->(1,12)
    (2,4)-->(2,12)
    (6,0)-->(6,12)
    (7,0)-->(7,11)
    (12,0)-->(12,6)'''
    global count


    '''
    rij = positions[0][1]  # rij
    kolom = positions[0][0]  # kolom
    ownpos = (rij, kolom)
    ownori = orientations[0]
    working = init_board(board)
    '''
    loaded_model = setup_model()

    working = init_board(board)
    inputlayer = get_inputlayer(working, orientations, positions)
    outputlayer = loaded_model.predict(inputlayer)
    ownposition = (positions[0][1], positions[0][0])
    print(ownposition, 'ownpos')
    ownorientation = orientations[0]
    # print(outputlayer, 'outl')
    # print('Neural position start', ownposition)

    move = isvalid_largest(working, ownposition, ownorientation, outputlayer)
    return move

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

def predict_next_pos(pos, ori, move):  # slang heeft pos en ori, volgende beweging is naar vaste plaats afh van move
    # move bepalen op basis van volgende plaats
    # ori 0-->5 moet naar diff gaan, om newpos te berekenen
    # correct!!!
    newori = (move + ori) % 6
    newpos = tuple(map(operator.add, pos, diff[newori]))
    print("newpos", newpos)
    return newpos

def setup_model():
    ar = np.load('Nets.npy')
    w0, w1, w2 = get_weights(ar)
    model = create_model(w0, w1, w2)
    return model




def get_weights(arr):
    '''Input is array (175, 261)
        Moet naar 3 arrays : (175, 128), (128,128) en (128,5)
        Nullen werden onderaan matrices toegevoegd, wegslicen'''

    w0 = arr[:,:128]
    w1 = arr[:128,128:256]
    w2 = arr[:128,256:]
    return w0, w1, w2

def create_model(w0, w1, w2):
    '''Returned model met weight matrices die input zijn'''
    zero128 = np.zeros((128))
    zero5 = np.zeros((5))

    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model =  model_from_json(loaded_model_json)

    model.layers[0].set_weights([w0, zero128])
    model.layers[1].set_weights([w1, zero128])
    model.layers[2].set_weights([w2, zero5])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

'''# test with 3 item highest in outputl
    maxout = 2
    move = moves[maxout]
    print('moveN', move)
    newori = (ori + move) % 6
    print(newori, 'neworiN')'''