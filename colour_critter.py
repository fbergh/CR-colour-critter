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

COLOURS = ["GREEN", "RED", "BLUE", "MAGENTA", "YELLOW"]


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

dt = 0.001
# #neurons for ensemble
N = 512
# #dimensions for SPA states
D = 256
# Threshold for the dot-product between colour and memory to see if it's counted
COLOUR_PERCEPTION_THRESH = 0.5
# Threshold for the difference between #goal_colours and #colours_found
STOP_THRESHOLD = 0.5
# Memory constant
MEMORY_CONSTANT = 1.5
# Exploration threshold
EXPLORE_THRESHOLD = 0.3
COOLDOWN_THRESHOLD = 0.35

#
CORRIDOR_DIST = 1.5

# Movement constants
EXPLORATION_TURN = 0.75
MAX_SPEED = 8.
MAX_STANDARD_TURN = 10.

# Vocabulary of colours
vc = spa.Vocabulary(D)
vc.parse('+'.join(COLOURS))

# Cooldown vocab
cooldown_vocab = spa.Vocabulary(D)
cooldown_vocab.parse('COOLDOWN')

# Your model might not be a nengo.Network() - SPA is permitted
model = spa.SPA()
with model:
    env = grid.GridNode(world, dt=0.001)

    # Three sensors for distance to the walls
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
    stim_radar = nengo.Node(detect)

    # radius=4, because angles are in range [0,4)
    radar = nengo.Ensemble(n_neurons=N, dimensions=3, radius=4)
    nengo.Connection(stim_radar[:3], radar)
    
    def compute_speed(sensor_distances):
        """
        Basic movement function that avoids walls
        based on distance to wall from front sensor
        """
        left, mid, right = sensor_distances
        return mid - 0.5


    def compute_turn(sensor_distances):
        """
        Basic movement function that avoids walls
        based on distance to wall between right and left sensor
        """
        left, mid, right = sensor_distances
        turn = right - left
        return turn
        
        
    def movement_func(t, x):
        speed, turn = x
        turn = turn * dt * MAX_STANDARD_TURN
        forward = speed * dt * MAX_SPEED
        
        # Perform action
        body.turn(turn)
        body.go_forward(forward)

    move = nengo.Node(movement_func, size_in=2)
    nengo.Connection(radar, move[0], function=compute_speed)
    nengo.Connection(radar, move[1], function=compute_turn)

    # Add noise
    noise_process = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 2), scale=False)
    noise_node = nengo.Node(noise_process)
    noise_ensemble = nengo.Ensemble(N, dimensions=1, noise=noise_process)
    nengo.Connection(noise_node, noise_ensemble)

    is_left_corridor_near = nengo.Ensemble(N, dimensions=1)
    is_right_corridor_near = nengo.Ensemble(N, dimensions=1)
    is_corridor_near_collector = nengo.Ensemble(N, dimensions=2)
    is_no_corridor_near = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(stim_radar[3], is_left_corridor_near, function=lambda distance: distance > 2)
    nengo.Connection(stim_radar[2], is_right_corridor_near, function=lambda distance: distance > 2)
    nengo.Connection(is_left_corridor_near, is_corridor_near_collector[0])
    nengo.Connection(is_right_corridor_near, is_corridor_near_collector[1])
    nengo.Connection(is_corridor_near_collector, is_no_corridor_near, function=lambda x: np.all(x < 0.5) > 0.8)
    nengo.Connection(is_no_corridor_near, noise_ensemble.neurons, transform=[[-10]] * N)

    # Cooldown memory to remember when we are allowed to turn
    model.cooldown_mem = spa.State(D, vocab=cooldown_vocab, feedback=0.9)
    model.cooldown_mem.output.output = lambda t, x: x

    def activate_cooldown(t, x):
        """ This function starts a cooldown if we have a noise spike and there is currently no cooldown """
        noise, is_no_cooldown_val = x
        cooldown_vector = np.zeros(D)
        is_no_cooldown_bool = True#is_no_cooldown_val > 0.5
        if is_no_cooldown_bool and (noise < -EXPLORE_THRESHOLD or noise > EXPLORE_THRESHOLD):
            cooldown_vector = cooldown_vocab["COOLDOWN"].v.reshape(D)
        # High constant because noise is only above threshold for a short duration
        return 50 * cooldown_vector

    # Hack: make intermediate cooldown node, because connection from noise to cooldown with a function resulted in
    # very weird noise values (perhaps due to noise??)
    intermediate_cooldown = nengo.Node(activate_cooldown, size_in=2, size_out=D)
    nengo.Connection(noise_ensemble, intermediate_cooldown[0])
    nengo.Connection(intermediate_cooldown, model.cooldown_mem.input)

    # Create an ensemble to save the value of the semantic pointer COOLDOWN which is used to inhibit restarting the
    # cooldown in intermediate_cooldown and exploring in movement_info
    cooldown_value = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(model.cooldown_mem.output, cooldown_value,
                     function=lambda cooldown_mem: np.dot(cooldown_mem, cooldown_vocab["COOLDOWN"].v.reshape(D)))
    nengo.Connection(cooldown_value, intermediate_cooldown[1])

    is_cooldown = nengo.Ensemble(N, dimensions=1)
    nengo.Connection(cooldown_value, is_cooldown, function=lambda cd_value: cd_value >= COOLDOWN_THRESHOLD)
    nengo.Connection(is_cooldown, noise_ensemble.neurons, transform=[[-10]] * N)
    nengo.Connection(is_cooldown, intermediate_cooldown[1])

    def exploration_move(t, x):
        """
        Determine agent rotation such that the agent will randomly go right
        or left based on a noisy signal and whether there is room to turn,
        as to explore the environment.
        """
        left_near_value, right_near_value, noise = x

        exploration_turn = 0
        if noise < -EXPLORE_THRESHOLD or noise > EXPLORE_THRESHOLD:
            if left_near_value > 0.1 and right_near_value > 0.1:
                if np.random.rand() > 0.5:
                    exploration_turn = -EXPLORATION_TURN
                    print(t, "RANDOM LEFT")
                else:
                    exploration_turn = EXPLORATION_TURN
                    print(t, "RANDOM RIGHT")
            elif left_near_value > 0.1:
                exploration_turn = -EXPLORATION_TURN
                print(t, "LEFT EXPLORE")
            else:
                exploration_turn = EXPLORATION_TURN
                print(t, "RIGHT EXPLORE")

        body.turn(exploration_turn)

    exploration = nengo.Node(exploration_move, size_in=3)
    nengo.Connection(is_left_corridor_near, exploration[0])
    nengo.Connection(is_right_corridor_near, exploration[1])
    nengo.Connection(noise_ensemble, exploration[2])

    # This node returns the colour of the cell currently occupied.
    # Note that you might want to transform this into something else
    # (see the assignment)
    current_color = nengo.Node(lambda t: body.cell.cellcolor)

    def recognise_colour(c):
        """ Send word from vocab if it's active (multiplied with a constant so it gets remembered) """
        vc_c = np.zeros(D)
        if c == 1:
            vc_c = vc["GREEN"].v.reshape(D)
        elif c == 2:
            vc_c = vc["RED"].v.reshape(D)
        elif c == 3:
            vc_c = vc["BLUE"].v.reshape(D)
        elif c == 4:
            vc_c = vc["MAGENTA"].v.reshape(D)
        elif c == 5:
            vc_c = vc["YELLOW"].v.reshape(D)
        return MEMORY_CONSTANT * vc_c

    # Create one big memory connected to all intermediate memories (add small feedback so it doesn't forget)
    model.all_mem = spa.State(D, vocab=vc, feedback=0.5)
    # Hack to make memory return its input
    model.all_mem.output.output = lambda t, x: x
    nengo.Connection(current_color, model.all_mem.input, function=recognise_colour)

    model.cleanup = spa.AssociativeMemory(input_vocab=vc, wta_output=False)
    nengo.Connection(model.cleanup.output, model.all_mem.input, synapse=0.05)
    nengo.Connection(model.all_mem.output, model.cleanup.input, synapse=0.05)

    # If colour_counter is an ensemble, remove t parameter
    def count_colours_from_memory(all_mem_output):
        """ Computes how many colours we have found based on the agent's memory """
        is_colour = np.array([np.dot(all_mem_output, vc[c].v.reshape(D))
                              for c in COLOURS])
        return np.sum(is_colour > COLOUR_PERCEPTION_THRESH)


    # Define a colour counter
    colour_counter = nengo.Ensemble(N, dimensions=1, radius=5)
    # # Connect it to the memory's output with the count_colours function
    nengo.Connection(model.all_mem.output, colour_counter,
                     function=count_colours_from_memory)

    # Node for user to input the desired amount of colours to find
    colour_goal = nengo.Node([0])
    # Ensemble to gather information about the goal number of colours and
    # how many colours we have already found
    colour_info = nengo.Ensemble(n_neurons=N, dimensions=2, radius=5)
    nengo.Connection(colour_goal, colour_info[0])
    nengo.Connection(colour_counter, colour_info[1])

    compute_diff = nengo.Ensemble(n_neurons=N, dimensions=1, radius=5)
    # Lambda x, where c_info[0] = goal, c_info[1] = counter
    nengo.Connection(colour_info, compute_diff, function=lambda c_info: c_info[0] - c_info[1])
    stop = nengo.Ensemble(n_neurons=N, dimensions=1, radius=1)
    nengo.Connection(compute_diff, stop, function=lambda difference: difference < STOP_THRESHOLD)
    nengo.Connection(stop, radar.neurons, transform=[[-10]] * N)
    nengo.Connection(stop, noise_ensemble.neurons, transform=[[-10]] * N)
