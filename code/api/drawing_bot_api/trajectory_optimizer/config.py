# model parameters ####################################
INPUT_DIM =                     10
NUM_LEADING_POINTS =            3
ACTION_DIM =                    2
HIDDEN_LAYER_DIM_ACTOR =        128
HIDDEN_LAYER_DIM_CRITIC =       256

TRANSFORMER_CRITIC_DIM =        2000

# Traing parameters ###################################
REWARD_DISCOUNT =               0.9
LR_CRITIC =                     0.00001
LR_ACTOR =                      0.0001

# Exploration settings #################################
OUTPUT_SCALING =                1

RANDOM_ACTION_PROBABILITY =     0
RANDOM_ACTION_DECAY =           0.996
RANDOM_ACTION_SCALE =           0.5

SIGMA_MIN =                     0.0001
SIGMA_MAX =                     1
SIGMA_INIT_WEIGHT_LIMIT =       0.05
SIGMA_OUTPUT_SCALING =          0.3
SIGMA_TRUE_SCALING =            0.05

SIGMA_ENTROPY_FACTOR =          0#.0005 # positive values enforce larger sigmas, negative values penalize larger sigmas
SIGMA_PENALTY_FACTOR =          0
ACTION_PENALTY_FACTOR =         0.05
ADVANTAGE_FACTOR =              0

GRADIENT_CLIPPING_LIMIT =       2

# Options ##############################################
VERBOSE =                       0
NUM_OF_CYCLES =                 1500

USE_PHASE_DIFFERENCE =          False
NORMALIZE_STATES =              True
COMBINE_STATES_FOR_CRITIC =     True
ADD_PROGRESS_INDICATOR =        True

SAVE_IMAGE_FREQ =               5
SAVE_SIMPLIFIED =               False

REWARD_DISTANCE_CLIPPING =      10
REWARD_NORMALIZATION_MODE =     'sigmoid' # options: 'linear', 'sigmoid'
GRANULAR_REWARD =               False
GRANULAR_REWARD_RESOLUTION =    25
SPARSE_REWARDS =                False
CUMULATIVE_VALUE_TRAINING =     True
STEP_WISE_REWARD =              False

TRAINABLE_SIGMA =               True
TRANSFORMER_CRITIC =            False
REWARD_LABELING =               True
