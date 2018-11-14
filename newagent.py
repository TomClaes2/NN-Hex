import numpy as np


compass = ['NW', 'NE', 'E', 'SE', 'SW', 'W']
working = None
count = 0
def generate_move(board, positions, orientations):
    '''array vn de vorm y,x
    playing field: (0,6)-->(0,12)
    (1,5)-->(1,12)
    (2,4)-->(2,12)
    (6,0)-->(6,12)
    (7,0)-->(7,11)
    (12,0)-->(12,6)'''
    global count


    rij = positions[0][1]  # rij
    kolom = positions[0][0]  # kolom

    working = init_board(board)
    print(working)
    #print(rij, kolom, 'coord')
    #print(orientations, 'ori')
    #print(positions, 'pos')
    move = np.random.randint(-2,3)
    move = 0
    if count %2 == 0:
        count += 1
        return 0
    count += 1
    if is_move_valid(working, move) is True:
        return 1
    return 0

def init_board(board):
    res = np.zeros((13,13))
    newb = board[:,:,0]+board[:,:,1]
    for i in range(6):
        res[i,5-i] = 1
    for i in range(6):
        res[i+7,12-i] = 1
    return res+newb

def is_move_valid(board, move):
    move = move
    newmove = (1,1)  # is vn de vorm [rij, kolom]
    if board[newmove] == 1:
        return False

    try:
        _ = board[newmove]
        return True
    except:
        return False



