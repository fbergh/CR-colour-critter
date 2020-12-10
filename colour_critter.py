import grid

mymap = """
#######
#  M  #
# # #B#
# # # #
#G Y R#
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
N = 500
# #dimensions for SPA states
D = 64
# Threshold for the dot-product between colour and memory to see if it's counted
C_THRESH = 0.5

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
        return [body.detect(d, max_distance=4)[0] for d in angles]


    stim_radar = nengo.Node(detect)

    # radius=4, because angles are in range [0,4)
    radar = nengo.Ensemble(n_neurons=N, dimensions=3, radius=4)
    nengo.Connection(stim_radar, radar)


    def movement_func(sensor_distances):
        """
        Basic movement function that avoids walls
        based on distance to wall between right and left sensor
        """
        left, mid, right = sensor_distances
        turn = right - left
        spd = mid - 0.5
        return spd, turn


    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)


    # Movement node gets as input speed and turn variables and uses these
    # to compute how much the agent should move and turn
    movement = nengo.Node(move, size_in=2)
    nengo.Connection(radar, movement, function=movement_func)

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

    # Create intermediate memory for every colour
    model.gre_mem = spa.State(D, feedback=1, vocab=vc)
    model.red_mem = spa.State(D, feedback=1, vocab=vc)
    model.blu_mem = spa.State(D, feedback=1, vocab=vc)
    model.mag_mem = spa.State(D, feedback=1, vocab=vc)
    model.yel_mem = spa.State(D, feedback=1, vocab=vc)
    # Connect current_color to all intermediate memories
    # (*3 such that seen values have a high value)
    nengo.Connection(current_color, model.gre_mem.input,
                     function=lambda c: 3 * int(c == 1) * vc["GREEN"].v.reshape(D))
    nengo.Connection(current_color, model.red_mem.input,
                     function=lambda c: 3 * int(c == 2) * vc["RED"].v.reshape(D))
    nengo.Connection(current_color, model.blu_mem.input,
                     function=lambda c: 3 * int(c == 3) * vc["BLUE"].v.reshape(D))
    nengo.Connection(current_color, model.mag_mem.input,
                     function=lambda c: 3 * int(c == 4) * vc["MAGENTA"].v.reshape(D))
    nengo.Connection(current_color, model.yel_mem.input,
                     function=lambda c: 3 * int(c == 5) * vc["YELLOW"].v.reshape(D))

    # Create one big memory connected to all intermediate memories
    model.all_mem = spa.State(D, feedback=1, vocab=vc)
    actions = spa.Actions(
        "all_mem = gre_mem + red_mem + blu_mem + mag_mem + yel_mem"
    )
    # spa.Cortical always carries out its action (unlike BasalGanglia)
    model.cort = spa.Cortical(actions=actions)

    # NOTE: Little hacky: all_mem.output is a Node that returns no output
    # By setting the node's output to the identity function, we can
    # access its memory (Do we actually want this?)
    model.all_mem.output.output = lambda t, x: x


    # If colour_counter is an ensemble, remove t parameter
    def count_colours_from_memory(t, all_mem):
        is_colour = np.array([np.dot(all_mem, vc[c].v.reshape(D))
                              for c in COLOURS])
        return np.sum(is_colour > C_THRESH)


    # Define a colour counter
    # NOTE: Should this be a node or an ensemble??
    colour_counter = nengo.Node(count_colours_from_memory, size_in=64)
    nengo.Connection(model.all_mem.output, colour_counter)
    # colour_counter = nengo.Ensemble(N, dimensions=1, radius=5)
    # # Connect it to the memory's output with the count_colours function
    # nengo.Connection(model.all_mem.output, colour_counter,
    #                  function=count_colours_from_memory)
