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

    # Generate replay
    with open('html.template', 'r', encoding='utf-8') as infile:
        template = infile.read()

    output = template.replace(
        '{{GENERATED_TRAJECTORY}}',
        json.dumps([p.trajectory for p in game.players]))

    with open('replay.html', 'w', encoding='utf-8') as outfile:
        outfile.write(output)

    print('A replay of the simulation has been written to "replay.html".')


def main(argv):
    """Script starting point.

    Args:
        argv (list): List of command-line arguments. The first element is
            always "simulator.py", the name of this script. The second element
            should be the name of the first agent's file. The final element
            should be the name of the opponent's file. Not that this can be
            the same name as the previous file.
    """

    if len(argv) != 3:
        print('Usage: python[3] simulator.py [agent_one] [agent_two]')
        exit(1)

    agent_one_file = argv[1]
    agent_one_module, _ = os.path.splitext(agent_one_file)
    print('mod', agent_one_module)
    generator_one_module = importlib.import_module(agent_one_module)

    agent_two_file = argv[2]
    agent_two_module, _ = os.path.splitext(agent_two_file)
    generator_two_module = importlib.import_module(agent_two_module)

    print('Running a simulation with "{}" vs. "{}".'.format(
        agent_one_file, agent_two_file))

    run_simulation(
        generator_one_module.generate_move,
        generator_two_module.generate_move)


if __name__ == '__main__':
    main(sys.argv)
