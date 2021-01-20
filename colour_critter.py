import grid
# import random
import numpy as np

mymap = """
#######
#  M  #
# # #B#
# #Y# #
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
EXPLORE_THRESHOLD = 0.6

# Vocabulary of colours
vc = spa.Vocabulary(D)
vc.parse('+'.join(COLOURS))

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
        arms = (np.linspace(-1, 1, 2) + body.dir) % world.directions
        angles = np.append(angles, arms)
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)

    # radius=4, because angles are in range [0,4)
    radar = nengo.Ensemble(n_neurons=N, dimensions=5, radius=4)
    nengo.Connection(stim_radar, radar)
    
    def compute_speed(sensor_distances):
        """
        Basic movement function that avoids walls
        based on distance to wall from front sensor
        """
        left, mid, right, arm_left, arm_right = sensor_distances
        return mid - 0.5


    def compute_turn(sensor_distances):
        """
        Basic movement function that avoids walls
        based on distance to wall between right and left sensor
        """
        left, mid, right, arm_left, arm_right = sensor_distances
        turn = right - left
        return turn, arm_left, arm_right

    
    def exploration_move(rotation, dt, max_rotate, do_move, noise, left_arm, right_arm, turn=0.75):
        """
        Determine agent rotation such that the agent will randomly go right
        or left based on a noisy signal, as to explore the
        environment.
        """
        avoid = rotation * dt * max_rotate
        move = avoid
        if (noise > EXPLORE_THRESHOLD) & (right_arm > 2):
            move = turn # right
        if (noise < -EXPLORE_THRESHOLD) & (left_arm > 2):
            move = -turn # left
        
        return move * do_move
        
        
    def move(t, x):
        speed, rotation, left_arm, right_arm, do_stop, noise = x
        dt = 0.001
        max_speed = 8.0
        max_rotate = 10.0
        # The agent should keep moving if it shouldn't stop (so invert do_stop)
        do_move = do_stop < 0.5
        # Compute rotation and speed 
        turn = exploration_move(rotation, dt, max_rotate, do_move, noise, left_arm, right_arm)
        forward = speed * dt * max_speed * do_move
        # Perform action
        body.turn(turn)
        body.go_forward(forward)

    # Create ensemble to gather information about the agent's movement
    # Namely: speed (dim 0), turn speed (dim 1), and if we should stop moving (dim 2)
    movement_info = nengo.Node(output=move, size_in=6)
    nengo.Connection(radar, movement_info[0], function=compute_speed)
    nengo.Connection(radar, movement_info[1:4], function=compute_turn)
    
    # Add noise
    noise_process = nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(0, 1), scale=False)
    noise = nengo.Node(noise_process)
    nengo.Connection(noise, movement_info[5])
    
    # if you wanted to know the position in the world, this is how to do it
    # The first two dimensions are X,Y coordinates, the third is the orientation
    # (plotting XY value shows the first two dimensions)
    def position_func(t):
        x_pos = body.x / world.width * 2 - 1
        y_pos = 1 - body.y / world.height * 2
        orientation = body.dir / world.directions
        return x_pos, y_pos, orientation
    position = nengo.Node(position_func)

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
    nengo.Connection(compute_diff, movement_info[4], function=lambda difference: difference < STOP_THRESHOLD)