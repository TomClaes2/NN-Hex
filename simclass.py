import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import operator

import simn
import NeuralAgent


class Simulate():
    def __init__(self):
        self.netar = self.load_array()
        json_file = open('models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.modelgen = model_from_json(loaded_model_json)
        self.zero128 = np.zeros((128))
        self.zero5 = np.zeros((5))
        self.losers = []


    def run_sim(self):
        for i in range(0, len(os.listdir("models/"))-1, 2):
            json_file = open('models/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model1 = model_from_json(loaded_model_json)
            model2 = model_from_json(loaded_model_json)

            model1.load_weights("models/model%s.h5" % (str(i)))
            model2.load_weights("models/model%s.h5" % (str(i+1)))

            #print(model1.layers[0].get_weights()[0][0], model1.layers[0].get_weights()[0][0])
            #if model1.layers[0].get_weights()[0][0] == model1.layers[0].get_weights()[0][0]:
             #   print('test')

            model1 = model1
            model2 = model2

            winner = self.battle()
        return winner

    def new_sim(self):
        loop = True
        i = 0
        length = self.netar.shape[0]
        while loop:
            array = self.netar[i*175:(i+1)*175][:]
            array2 = self.netar[(i+1)*175:(i+2)*175,:]

            w0, w1, w2 = self.get_weights(array)
            w02, w12, w22 = self.get_weights(array2)

            self.model1 = self.create_model(w0, w1, w2)
            self.model2 = self.create_model(w02, w12, w22)

            loser = self.battle() # is altijd 0 of 1

            self.losers.append(i+loser)

            i += 2
            if i == int(length/175):
                loop = False
        self.clean_array(self.losers)
        print(self.save_array(), 'shapenet' )


    def load_array(self):
        ar = np.load('Nets.npy')
        return ar

    def save_array(self):
        np.save('Nets.npy', self.netar)
        return self.netar.shape

    def clean_array(self, losers):
        '''Losers is lijst met verloren nets die moeten verwijderd worden uit de array vooraleer opnieuw
        opgeslagen wordt.
        elementen zijn integers, met e= het e+1-ste net in de array
        itereren in andere richting zodat indexen niet veranderen

        np.delete(arr, (start,stop), axis=0) delete alle rijen incluus start en stop(indexen, start bij 0)'''
        for e in losers[::-1]:
            self.netar = np.delete(self.netar, slice(e*175, (e+1)*175), axis=0)

        return None



    def get_weights(self, arr):
        '''Input is array (175, 261)
        Moet naar 3 arrays : (175, 128), (128,128) en (128,5)
        Nullen werden onderaan matrices toegevoegd, wegslicen'''

        w0 = arr[:,:128]
        w1 = arr[:128,128:256]
        w2 = arr[:128,256:]
        return w0, w1, w2

    def create_model(self, w0, w1, w2):
        '''Returned model met weight matrices die input zijn'''
        model = self.modelgen

        model.layers[0].set_weights([w0, self.zero128])
        model.layers[1].set_weights([w1, self.zero128])
        model.layers[2].set_weights([w2, self.zero5])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


    def battle(self):
        '''Moet 0 of 1 returnen op basis van wie match verloren heeft, bij draw kies randint(0,1)
        status: lijst met 2 waarden status player1, player 2:
        0: geldige positie
        1: tegen muur gebotst
        2: tegen staart gebotst'''
        status = simn.main(self.generate_move1, self.generate_move2)
        if status[0] != 0 and status[1] !=0:
            return np.random.randint(0,1)
        if status[0] != 0:
            return 0
        if status[1] != 0:
            return 1

    def gen_inputlayer(self):
        rand = np.random.uniform(0,1, size=175)
        rand = rand.reshape((1,175))
        return rand

    def generate_move1(self, board, pos, ori):
        working = NeuralAgent.init_board(board)
        inputlayer = NeuralAgent.get_inputlayer(working, ori, pos)
        outputlayer = self.model1.predict(inputlayer)

        ownposition = (pos[0][1],pos[0][0])
        ownorientation = ori[0]

        move = NeuralAgent.isvalid_largest(working, ownposition, ownorientation, outputlayer)

        return move

    def generate_move2(self, board, pos, ori):
        working = NeuralAgent.init_board(board)
        inputlayer = NeuralAgent.get_inputlayer(working, ori, pos)
        outputlayer = self.model2.predict(inputlayer)

        ownposition = (pos[0][1], pos[0][0])
        ownorientation = ori[0]

        move = NeuralAgent.isvalid_largest(working, ownposition, ownorientation, outputlayer)

        return move



# weight = model.layers[h].get_weights()[0]

sim = Simulate()
print(sim.new_sim())

