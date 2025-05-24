# model parameters ####################################
INPUT_DIM =                     10
NUM_LEADING_POINTS =            3
ACTION_DIM =                    2
HIDDEN_LAYER_DIM_ACTOR =        128
HIDDEN_LAYER_DIM_CRITIC =       128

# Traing parameters ###################################
REWARD_DISCOUNT =               1
LR =                            0.001

# Exploration settings #################################
OUTPUT_SCALING =                3

# sigma settings
SIGMA_MIN =                     0.01
SIGMA_MAX =                     0.12
SIGMA_INIT_WEIGHT_LIMIT =       1
SIGMA_OUTPUT_SCALING =          0.1
SIGMA_TRUE_SCALING =            50

# settings for custom loss #############################
SIGMA_ENTROPY_FACTOR =          0 # positive values enforce larger sigmas, negative values penalize larger sigmas
SIGMA_PENALTY_FACTOR =          0
ACTION_PENALTY_FACTOR =         0 #.0001
ADVANTAGE_FACTOR =              0
ACTION_LEAK =                   0
GRADIENT_CLIPPING_LIMIT =       100

# Options ##############################################
VERBOSE =                       0
NUM_OF_CYCLES =                 300

USE_PHASE_DIFFERENCE =          False
ADD_PROGRESS_INDICATOR =        True

SAVE_IMAGE_FREQ =               1
SAVE_ORIGINAL =               False

REWARD_DISTANCE_CLIPPING =      10
REWARD_NORMALIZATION_MODE =     'sigmoid' # options: 'linear', 'sigmoid'
GRANULAR_REWARD =               True
GRANULAR_REWARD_RESOLUTION =    3
SPARSE_REWARDS =                False
CUMULATIVE_VALUE_TRAINING =     True