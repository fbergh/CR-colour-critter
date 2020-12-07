import grid

mymap = """
#######
#  M  #
# # # #
# #B# #
#G Y R#
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

N = 500
D = 32

import nengo
import nengo.spa as spa
import numpy as np

vc = spa.Vocabulary(D)
vc.parse("GREEN+RED+BLUE+MAGENTA+YELLOW")

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
    
    # colour_memory = nengo.Ensemble(n_neurons=N, dimensions=5)
    # # Green
    # nengo.Connection(current_color, colour_memory[0], function=lambda c: c==1)
    # # Red
    # nengo.Connection(current_color, colour_memory[1], function=lambda c: c==2)
    # # Blue
    # nengo.Connection(current_color, colour_memory[2], function=lambda c: c==3)
    # # Magenta
    # nengo.Connection(current_color, colour_memory[3], function=lambda c: c==4)
    # # Yellow
    # nengo.Connection(current_color, colour_memory[4], function=lambda c: c==5)
    
    model.gre_mem = spa.State(D, feedback=1, vocab=vc)
    model.red_mem = spa.State(D, feedback=1, vocab=vc)
    model.blu_mem = spa.State(D, feedback=1, vocab=vc)
    model.mag_mem = spa.State(D, feedback=1, vocab=vc)
    model.yel_mem = spa.State(D, feedback=1, vocab=vc)
    # def fun(x):
    #     print(x)
    #     return np.ones((D,1)) * x
    nengo.Connection(current_color, model.gre_mem.input,
                     transform=vc["GREEN"].v.reshape(D,1))
    # nengo.Connection(current_color, model.red_mem.input,
    #                  function=lambda c: c==2 * vc["RED"].v.reshape(D,1))
    # nengo.Connection(current_color, model.blu_mem.input,
    #                  function=lambda c: c==3 * vc["BLUE"].v.reshape(D,1))
    # nengo.Connection(current_color, model.mag_mem.input,
    #                  function=lambda c: c==4 * vc["MAGENTA"].v.reshape(D,1))
    # nengo.Connection(current_color, model.yel_mem.input,
    #                  function=lambda c: c==5 * vc["YELLOW"].v.reshape(D,1))
    # nengo.Connection(colour_memory[0], 
    #                  model.c_mem.input, 
    #                  transform=vc["GREEN"].v.reshape(D, 1))
    # nengo.Connection(colour_memory[1], 
    #                  model.c_mem.input, 
    #                  transform=vc["RED"].v.reshape(D, 1))
    # nengo.Connection(colour_memory[2], 
    #                  model.c_mem.input, 
    #                  transform=vc["BLUE"].v.reshape(D, 1))
    # nengo.Connection(colour_memory[3], 
    #                  model.c_mem.input, 
    #                  transform=vc["MAGENTA"].v.reshape(D, 1))
    # nengo.Connection(colour_memory[4], 
    #                  model.c_mem.input, 
    #                  transform=vc["YELLOW"].v.reshape(D, 1))
    # nengo.Connection(colour_memory, colour_memory, transform=2)
    
    # colour_counter = nengo.Ensemble(n_neurons=N, dimensions=1, radius=5)
    # def thresh_sum(c_memory):
    #     counter = []
    #     for c in c_memory:
    #         counter.append(c > 0.8)
    #     return sum(counter)
    # nengo.Connection(colour_memory, colour_counter, function=thresh_sum)
    
    