import grid
# import random
import numpy as np

mymap = """
#######
#  M  #
#    B#
#  Y  #
#G   R#
#######
"""


class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'

        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True

        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5


world = grid.World(Cell, map=mymap, directions=4)

body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

import nengo
import nengo.spa as spa
import numpy as np
from config import *

# Colour vocabulary
COLOURS = ["GREEN", "RED", "BLUE", "MAGENTA", "YELLOW"]
colour_vocab = spa.Vocabulary(D)
colour_vocab.parse('+'.join(COLOURS))
# Cooldown vocab
cooldown_vocab = spa.Vocabulary(D)
cooldown_vocab.parse('COOLDOWN')

### FUNCTIONS ###
def detect(t):
    """
    Returns distance from wall for each sensor (maximally 4)
    Angles of sensors are -0.5, 0, 0.5 relative to the body's angle
    Angles are in range [0,4) (1=east, 2=south, 3=west)
    """
    angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions

    # Added arms to be able to detect whether there is room to turn
    # left or right.
    arms = (np.linspace(-1, 1, 2) + body.dir) % world.directions
    angles = np.append(angles, arms)

    return [body.detect(d, max_distance=4)[0] for d in angles]


def compute_speed(sensor_distances):
    """
    Basic movement function that avoids walls
    based on distance to wall from front sensor
    """
    _, mid, _ = sensor_distances
    return mid - 0.5


def compute_turn(sensor_distances):
    """
    Basic movement function that avoids walls
    based on distance to wall between right and left sensor
    """
    left, _, right = sensor_distances
    turn = right - left
    return turn


def movement_func(t, x):
    """
    Movement function to determine how much the agent will turn and move forward this timestep
    Also incorporates the exploration turn, if it is above a certain threshold
    """
    speed, turn, exploration_val = x
    turn = turn * dt * MAX_STANDARD_TURN
    forward = speed * dt * MAX_SPEED

    # Explore right if we are above positive threshold and left if we are below negative threshold
    # t > 0.1 so the agent doesn't explore at initialisation
    if exploration_val > EXPLORATION_THRESHOLD and t > 0.1:
        body.turn(EXPLORATION_TURN)
        print("boe")
    elif exploration_val < -EXPLORATION_THRESHOLD and t > 0.1:
        body.turn(-EXPLORATION_TURN)
        print("schrik") 
    # Otherwise do a regular turn
    else:
        body.turn(turn)
    body.go_forward(forward)


def activate_cooldown(t, inputs):
    """
    We only activate the cooldown (i.e. we only spike) if
        1) there is currently no cooldown (is_cooldown);
        2) there is currently a corridor visible (is_corridor_near)
        3) the noise value is above the threshold (NOISE_SPIKE_THRESHOLD)
    """
    noise, is_cooldown_val, is_no_corridor_near_val = inputs
    cooldown_vector = np.zeros(D)
    is_cooldown_bool = is_cooldown_val > 0.5
    is_corridor_near_bool = is_no_corridor_near_val < 0.5
    if is_corridor_near_bool and not is_cooldown_bool \
            and (noise < -NOISE_SPIKE_THRESHOLD or noise > NOISE_SPIKE_THRESHOLD):
        cooldown_vector = cooldown_vocab["COOLDOWN"].v.reshape(D)
    # High constant because noise is only above threshold for a short duration
    return 50 * cooldown_vector


def exploration_move(x):
    """
    Determine if the agent should explore left or right given if
        1) there is a spike;
        2) there is a corridor to the left and/or right
    If there is a corridor on both sides, the agent chooses randomly between left or right
    """
    left_near_value, right_near_value, noise = x

    if noise < -NOISE_SPIKE_THRESHOLD or noise > NOISE_SPIKE_THRESHOLD:
        if left_near_value > CORRIDOR_THRESHOLD and right_near_value > CORRIDOR_THRESHOLD:
            # Random choice between left or right
            return -EXPLORATION_IMPULSE if np.random.rand() > 0.5 else EXPLORATION_IMPULSE
        elif left_near_value > CORRIDOR_THRESHOLD:
            return -EXPLORATION_IMPULSE
        else:
            return EXPLORATION_IMPULSE
    else:
        return 0


def recognise_colour(c):
    """ Send word from vocab if it's active (multiplied with a constant so that it gets remembered) """
    memory_vector = np.zeros(D)
    if c == 1:
        memory_vector = colour_vocab["GREEN"].v.reshape(D)
    elif c == 2:
        memory_vector = colour_vocab["RED"].v.reshape(D)
    elif c == 3:
        memory_vector = colour_vocab["BLUE"].v.reshape(D)
    elif c == 4:
        memory_vector = colour_vocab["MAGENTA"].v.reshape(D)
    elif c == 5:
        memory_vector = colour_vocab["YELLOW"].v.reshape(D)
    return MEMORY_CONSTANT * memory_vector


def count_colours_from_memory(all_mem_output):
    """ Computes how many colours we have found based on the agent's memory """
    # Use dot-product like SPA state to determine how active a certain colour is
    is_colour = np.array([np.dot(all_mem_output, colour_vocab[c].v.reshape(D))
                          for c in COLOURS])
    return np.sum(is_colour > COLOUR_PERCEPTION_THRESH)


### MODEL ###
# Your model might not be a nengo.Network() - SPA is permitted
model = spa.SPA()
with model:
    env = grid.GridNode(world, dt=0.001)

    ### SENSING ###
    # Stim_radar detects the distances from all its sensors (arms and eyes)
    stim_radar = nengo.Node(detect)

    ### CORRIDOR SENSING ###
    # Nengo ensembles to determine whether a corridor is near based on information from the arms
    # (received form stim_radar)
    is_right_corridor_near = nengo.Ensemble(N, dimensions=1)
    is_left_corridor_near = nengo.Ensemble(N, dimensions=1)
    is_corridor_near_collector = nengo.Ensemble(N, dimensions=2)
    nengo.Connection(stim_radar[3], is_left_corridor_near, function=lambda distance: distance > CORRIDOR_DIST)
    nengo.Connection(stim_radar[4], is_right_corridor_near, function=lambda distance: distance > CORRIDOR_DIST)
    nengo.Connection(is_left_corridor_near, is_corridor_near_collector[0])
    nengo.Connection(is_right_corridor_near, is_corridor_near_collector[1])
    # Create an ensemble to determine whether there is no corridor to the left or the right for inhibition later
    is_no_corridor_near = nengo.Ensemble(N, dimensions=1)
    # np.all(x < 0.5) computes whether there is no corridor near, > 0.8 is for robustness
    nengo.Connection(is_corridor_near_collector, is_no_corridor_near, function=lambda x: np.all(x < 0.5) > 0.8)

    ### NOISE ###
    # Create a noise process which is used for whether the agent should explore
    noise_process = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 2), scale=False)
    noise_node = nengo.Node(noise_process)

    ### COOLDOWN SYSTEM ###
    # Create a memory to remember when we are allowed to turn
    model.cooldown_mem = spa.State(D, vocab=cooldown_vocab, feedback=0.9)
    # Hack: make a SPA state return its input
    model.cooldown_mem.output.output = lambda t, x: x

    # Hack: make intermediate cooldown node, because an ensemble cannot properly parse noise
    intermediate_cooldown = nengo.Node(activate_cooldown, size_in=3, size_out=D)
    nengo.Connection(intermediate_cooldown, model.cooldown_mem.input)

    # Create an ensemble to save the activity of the semantic pointer COOLDOWN which is used to determine whether a
    # cooldown is currently active (with a similar dot product to colour memory)
    cooldown_value = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(model.cooldown_mem.output, cooldown_value,
                     function=lambda cooldown_mem: np.dot(cooldown_mem, cooldown_vocab["COOLDOWN"].v.reshape(D)))

    # Create an ensemble to determine whether a cooldown is active based on the cooldown_value ensemble
    is_cooldown = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(cooldown_value, is_cooldown, function=lambda cd_value: cd_value >= COOLDOWN_THRESHOLD)

    # Connect all information relevant fora cooldown to intermediate_cooldown (see activate_cooldown function)
    nengo.Connection(noise_node, intermediate_cooldown[0])
    nengo.Connection(is_cooldown, intermediate_cooldown[1])
    nengo.Connection(is_no_corridor_near, intermediate_cooldown[2])


    ### COLOUR COUNTING SYSTEM ###
    # Create node thatreturns the colour of the cell currently occupied
    current_color = nengo.Node(lambda t: body.cell.cellcolor)

    # Create a memory to remember which colours the agent has seen (add small feedback so it doesn't forget)
    model.colour_memory = spa.State(D, vocab=colour_vocab, feedback=0.5)
    model.colour_memory.output.output = lambda t, x: x
    nengo.Connection(current_color, model.colour_memory.input, function=recognise_colour)
    # Add an associative memory to further reinforce remembering
    # (wta_output=False because we want to remember multiple things at once)
    model.cleanup = spa.AssociativeMemory(input_vocab=colour_vocab, wta_output=False)
    nengo.Connection(model.cleanup.output, model.colour_memory.input, synapse=0.05)
    nengo.Connection(model.colour_memory.output, model.cleanup.input, synapse=0.05)

    # Create a colour counter that keeps track of the number of colours we have seen, based on the memory
    # There are 5 colours, so radius=5
    colour_counter = nengo.Ensemble(N, dimensions=1, radius=5)
    nengo.Connection(model.colour_memory.output, colour_counter, function=count_colours_from_memory)

    # Create node for user to input the desired amount of colours to find
    colour_goal = nengo.Node([0])
    # Create ensemble to gather information on the goal number of colours and how many colours we have already found
    colour_info = nengo.Ensemble(n_neurons=N, dimensions=2, radius=5)
    nengo.Connection(colour_goal, colour_info[0])
    nengo.Connection(colour_counter, colour_info[1])

    # Create ensemble to compute how many colours we still have left
    compute_diff = nengo.Ensemble(n_neurons=N, dimensions=1, radius=5)
    # c_info[0] = goal, c_info[1] = counter
    nengo.Connection(colour_info, compute_diff, function=lambda c_info: c_info[0] - c_info[1])
    # Create final ensemble to determine if we have found the goal number of colours in which case we can stop moving
    found_colours = nengo.Ensemble(n_neurons=N, dimensions=1, radius=1)
    nengo.Connection(compute_diff, found_colours, function=lambda difference: difference < STOP_THRESHOLD)

    ### MOVEMENT SYSTEM ###
    # REGULAR MOVEMENT (EYES) #
    # Regular movement is based on information from the eyes (radius=4, because angles are in range [0,4))
    radar = nengo.Ensemble(n_neurons=N, dimensions=3, radius=4)
    nengo.Connection(stim_radar[:3], radar)

    # EXPLORATION (ARMS) #
    # Intermediate ensemble to collect information relevant to whether the agent should explore
    # (see exploration_move function)
    exploration_collector = nengo.Ensemble(N, dimensions=3)
    nengo.Connection(is_left_corridor_near, exploration_collector[0])
    nengo.Connection(is_right_corridor_near, exploration_collector[1])
    nengo.Connection(noise_node, exploration_collector[2])

    # EXCITATION & INHIBITION #
    # Create an output node to determine how much movement should occur based on the speed and turn computed from the
    # radar and exploration
    move = nengo.Node(movement_func, size_in=3)
    nengo.Connection(radar, move[0], function=compute_speed)
    nengo.Connection(radar, move[1], function=compute_turn)
    nengo.Connection(exploration_collector, move[2], function=exploration_move)

    # If there is currently a cooldown or there is no corridor visible, we should not explore
    nengo.Connection(is_cooldown, exploration_collector.neurons, transform=[[INHIBITION_CONSTANT]] * N)
    nengo.Connection(is_no_corridor_near, exploration_collector.neurons, transform=[[INHIBITION_CONSTANT]] * N)
    # If we have found the target number of colours, we can stop all forms of movement
    nengo.Connection(found_colours, radar.neurons, transform=[[INHIBITION_CONSTANT]] * N)
    nengo.Connection(found_colours, exploration_collector.neurons, transform=[[INHIBITION_CONSTANT]] * N)
