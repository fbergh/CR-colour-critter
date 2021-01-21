# Nengo #
dt = 0.001
N = 512  # neurons in an ensemble
D = 256  # dimensions in SPA state

# Perception thresholds and constants #
# Threshold for the dot-product between colour and memory to see if it's counted
COLOUR_PERCEPTION_THRESH = 0.5
# Threshold for the difference between #goal_colours and #colours_found
STOP_THRESHOLD = 0.5
# Memory constant for initial colour activation in colour memory
MEMORY_CONSTANT = 1.5
# Noise threshold used to decide whether the agent should explore
NOISE_SPIKE_THRESHOLD = 0.7
# Due to short exploration impulse, return a high value so that it is usable
EXPLORATION_IMPULSE = 100
# Activation constant from exploration_collector to move, needed due to short exploration impulse
EXPLORATION_THRESHOLD = 12
# Threshold for when the cooldown is determined to be over
COOLDOWN_THRESHOLD = 0.35
# Threshold for when we perceive something as a corridor
CORRIDOR_DIST = 1.5
# Threshold to determine if the is_corridor_near boolean is large enough for there to be an actual corridor
CORRIDOR_THRESHOLD = 0.3

# Movement constants #
# The amount that the agent should turn if it wants to explore
EXPLORATION_TURN = 0.4
# Default max speed and turn
MAX_SPEED = 8.
MAX_STANDARD_TURN = 10.
INHIBITION_CONSTANT = -2