import importlib
import json
import os
import sys
import NeuralAgent

from game import Hexatron


def run_simulation(move_generator_one, move_generator_two):
    """Set up a single game, where two agents play each other.
    A replay will be written to "replay.html".

    Args:
        move_generator_one (function): The first agent's implementation for
            `generate_move`.
        move_generator_two (function): The opponents implementation for
            `generate_move`.
    """

    # Instantiate game
    game = Hexatron(13)
    observation = game.reset()

    # Play game
    done = False
    while not done:

        # Generate moves
        move1 = move_generator_one(
            observation['board'],
            observation['positions'],
            observation['orientations'])

        move2 = move_generator_two(
            observation['board'][:, :, ::-1],
            observation['positions'][::-1],
            observation['orientations'][::-1])

        observation, done, status = game.act(move1, move2)
    return status



def main(gen_move1, gen_move2): # twee argumenten zijn fies die move returnen
    """Script starting point.

    Args:
        argv (list): List of command-line arguments. The first element is
            always "simulator.py", the name of this script. The second element
            should be the name of the first agent's file. The final element
            should be the name of the opponent's file. Not that this can be
            the same name as the previous file.
    """

    print('Running a simulation with "{}" vs. "{}".'.format(
        'test', 'test2'))

    st = run_simulation(
        gen_move1,
        gen_move2)

    return st

if __name__ == '__main__':
    main(sys.argv)


