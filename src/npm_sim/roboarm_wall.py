from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.utils
import newton.viewer
import numpy as np
import warp as wp

from .materials import MATERIALS, MaterialPreset
from .rigid_ramp_tower import _build_viewer, _static_shape_cfg

# Fixed FR3/Panda robot-arm pushing demo.
# Axis convention:
#   z = up
#   y = push direction
#   x = sideways
FRAME_RATE = 120
SIM_SUBSTEPS = 10
SOLVER_ITERATIONS = 20
SOLVER_CONTACT_RELAXATION = 0.85
SOLVER_ANGULAR_DAMPING = 0.02
RIGID_GAP = 0.003

FLOOR_LENGTH = 2.80
FLOOR_WIDTH = 1.80
FLOOR_THICKNESS = 0.08
FLOOR_CENTER_Y = 0.80

WALL_WIDTH = 0.68
WALL_THICKNESS = 0.24
WALL_HEIGHT = 0.66
WALL_CENTER_Y = 0.72
WALL_DENSITY_SCALE = 0.50

FRANKA_ASSET_NAME = "franka_emika_panda"
FRANKA_URDF = "urdf/fr3_franka_hand.urdf"
FRANKA_BASE_POS = wp.vec3(0.0, -0.05, 0.0)
FRANKA_BASE_YAW_DEG = 90.0
FRANKA_HAND_LABEL = "fr3_hand"
FRANKA_TCP_LABEL = "fr3_hand_tcp"
FRANKA_DOF_COUNT = 9

# The tool is attached to fr3_hand, not to the massless TCP helper body.
TOOL_LOCAL_POS = wp.vec3(0.0, 0.0, -0.1034)
TOOL_RADIUS = 0.06

# The contact point follows this world-space line. The wall normal is +y/-y,
# so changing only y makes the hit horizontal and perpendicular to the wall.
PUSH_MODE_HIGH = "high"
PUSH_MODE_LOW = "low"
PUSH_MODE_CENTER = "center"
DEFAULT_PUSH_MODE = PUSH_MODE_HIGH
TOOL_PATH_X = 0.0
TOOL_PATH_START_Y = 0.35
TOOL_PATH_END_Y = 0.57
TOOL_PATH_Z_BY_MODE = {
    PUSH_MODE_HIGH: 0.54,
    PUSH_MODE_LOW: 0.12,
    PUSH_MODE_CENTER: WALL_HEIGHT * 0.5,
}
ROBOT_HOLD_SECONDS = 0.18
ROBOT_ACCEL_SECONDS = 0.50

HEIGHT_PUSH_FRICTION_MODE = "height-default"
FRICTION_MODE_HIGH = "high"
FRICTION_MODE_LOW = "low"
DEFAULT_FRICTION_MODE = HEIGHT_PUSH_FRICTION_MODE
FRICTION_COEFFICIENT_BY_MODE = {
    FRICTION_MODE_HIGH: 0.50,
    FRICTION_MODE_LOW: 0.20,
}

FRANKA_CENTER_TOOL_PATH_Q = np.array(
    [
        [-1.71250587e-08, 6.86784536e-02, -2.43200304e-08, -2.33256817e00, -3.07180992e-09, 5.96292913e-01, 7.02418301e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.29215373e-08, 8.26818720e-02, -8.12976140e-08, -2.31285405e00, -1.35136986e-08, 6.05346382e-01, 3.10340154e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.09787867e-08, 9.66617614e-02, -2.65587250e-08, -2.29305720e00, -2.97122682e-08, 6.14411175e-01, 3.71851772e-09, 3.99999991e-02, 3.99999991e-02],
        [-8.09657692e-08, 1.10624112e-01, -8.67590231e-08, -2.27317333e00, -9.91876270e-09, 6.23491943e-01, 4.82611862e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.17680088e-08, 1.24574631e-01, -2.66712590e-08, -2.25319910e00, 4.82654716e-09, 6.32594824e-01, 7.17457027e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.13864473e-11, 1.38519108e-01, 2.53020005e-09, -2.23313046e00, 2.04230872e-08, 6.41726673e-01, 1.84441129e-09, 3.99999991e-02, 3.99999991e-02],
        [1.35901104e-08, 1.52462944e-01, -7.80175924e-11, -2.21296406e00, 9.88923254e-09, 6.50889516e-01, 2.39222664e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.85603869e-08, 1.66411430e-01, -5.21246513e-08, -2.19269514e00, -2.66516342e-09, 6.60089612e-01, 9.20077126e-10, 3.99999991e-02, 3.99999991e-02],
        [-2.52546961e-08, 1.80370376e-01, -2.56252655e-08, -2.17231965e00, 6.23103080e-09, 6.69333637e-01, -5.82496940e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.71635342e-08, 1.94345534e-01, -8.05536402e-08, -2.15183306e00, 1.75841075e-09, 6.78625464e-01, 7.74995623e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.07349595e-08, 2.08342150e-01, -2.35764297e-08, -2.13123012e00, 3.65099617e-09, 6.87968016e-01, 1.24491688e-08, 3.99999991e-02, 3.99999991e-02],
        [9.40606881e-09, 2.22366050e-01, 1.62891212e-08, -2.11050653e00, 8.16589285e-09, 6.97371960e-01, -2.22429342e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.59814239e-08, 2.36422479e-01, -6.52715642e-08, -2.08965683e00, 8.43558468e-09, 7.06837595e-01, -1.72518488e-09, 3.99999991e-02, 3.99999991e-02],
        [2.80684791e-08, 2.50517786e-01, 1.31328015e-08, -2.06867528e00, -2.76381318e-09, 7.16371357e-01, 9.86605908e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.01385591e-08, 2.64656574e-01, -4.98113728e-08, -2.04755712e00, 3.93201294e-09, 7.25980282e-01, -4.51614879e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.57284469e-08, 2.78846174e-01, -3.49136116e-08, -2.02629447e00, -1.40952023e-08, 7.35668540e-01, 2.71678546e-09, 3.99999991e-02, 3.99999991e-02],
        [-8.25028081e-08, 2.93091238e-01, -9.35823721e-08, -2.00488329e00, 1.48255497e-08, 7.45443165e-01, 3.80030496e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.12649088e-08, 3.07398766e-01, -6.90983626e-08, -1.98331559e00, 1.12682521e-08, 7.55308568e-01, -2.63734790e-09, 3.99999991e-02, 3.99999991e-02],
        [1.27864510e-08, 3.21775377e-01, 2.55841215e-08, -1.96158409e00, 1.87765572e-08, 7.65271962e-01, 9.53279766e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.16524724e-08, 3.36226195e-01, -7.05411267e-08, -1.93968272e00, -9.32827060e-09, 7.75340199e-01, -3.60319707e-09, 3.99999991e-02, 3.99999991e-02],
        [-9.57891473e-08, 3.50767046e-01, -9.60536539e-08, -1.91759157e00, 1.22896227e-08, 7.85522580e-01, -6.24062446e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.23642845e-08, 3.65390360e-01, -6.51851622e-08, -1.89532161e00, 4.62124206e-09, 7.95816958e-01, -1.03550866e-08, 3.99999991e-02, 3.99999991e-02],
        [-6.24962695e-08, 3.80109042e-01, -7.20476834e-08, -1.87285697e00, 3.13587591e-08, 8.06239665e-01, -3.31395889e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.29946237e-08, 3.94932657e-01, -8.22800175e-08, -1.85018563e00, 2.30332020e-09, 8.16793025e-01, -4.68445860e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.20843566e-09, 4.09868181e-01, -4.19356549e-10, -1.82729852e00, 1.79583779e-08, 8.27486455e-01, -3.46115803e-09, 3.99999991e-02, 3.99999991e-02],
        [-8.34960616e-08, 4.24924165e-01, -8.28960367e-08, -1.80418420e00, 1.94759693e-08, 8.38329375e-01, -1.44867252e-09, 3.99999991e-02, 3.99999991e-02],
        [1.59857905e-09, 4.40109044e-01, -1.74843890e-08, -1.78083241e00, 1.79613231e-08, 8.49331081e-01, -2.12249840e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.67223060e-09, 4.55434322e-01, -1.59266893e-08, -1.75722766e00, 2.87225141e-08, 8.60498965e-01, -9.87655646e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.32842128e-08, 4.70908672e-01, -2.15819984e-08, -1.73335803e00, -1.73698051e-08, 8.71844590e-01, -1.20508394e-08, 3.99999991e-02, 3.99999991e-02],
        [-7.38014521e-08, 4.86542642e-01, -6.67227908e-08, -1.70920968e00, 5.38606137e-09, 8.83381069e-01, -3.60608365e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.10616768e-08, 5.02347887e-01, -3.31958283e-08, -1.68476677e00, 3.17005444e-08, 8.95117819e-01, -8.61266436e-09, 3.99999991e-02, 3.99999991e-02],
        [-8.75941311e-08, 5.18338323e-01, -9.45034131e-08, -1.66000962e00, 9.35726341e-09, 9.07068014e-01, -1.46791272e-08, 3.99999991e-02, 3.99999991e-02],
        [-5.67642005e-08, 5.34526110e-01, -5.37613865e-08, -1.63492179e00, 1.95562180e-08, 9.19245541e-01, -1.52629784e-08, 3.99999991e-02, 3.99999991e-02],
    ],
    dtype=np.float32,
)

# Re-solved IK paths for the fair comparison: both modes use the same
# y-trajectory from 0.35m to 0.57m, with only the target height changed.
FRANKA_HIGH_TOOL_PATH_Q = np.array(
    [
        [-4.80801887e-08, -2.85256803e-01, -5.28747073e-08, -2.15261245e00, -5.00378183e-09, 5.48033774e-01, 1.42425705e-09, 3.99999991e-02, 3.99999991e-02],
        [-6.18505425e-08, -2.66523123e-01, -6.91699142e-08, -2.13463688e00, -6.37534336e-09, 5.57897985e-01, 1.36989642e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.44032101e-08, -2.47889161e-01, -2.76532433e-08, -2.11650181e00, 3.13139537e-09, 5.67786992e-01, 1.17057553e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.26776347e-08, -2.29347825e-01, -5.93059042e-08, -2.09820580e00, -2.08343476e-09, 5.77706337e-01, 5.14188470e-10, 3.99999991e-02, 3.99999991e-02],
        [-8.26119795e-09, -2.10891470e-01, -1.14199139e-08, -2.07974720e00, 7.20402138e-09, 5.87661564e-01, 1.09864493e-10, 3.99999991e-02, 3.99999991e-02],
        [-6.32273824e-08, -1.92512676e-01, -7.01487153e-08, -2.06112456e00, -1.80357684e-09, 5.97658396e-01, -2.00898118e-10, 3.99999991e-02, 3.99999991e-02],
        [-3.79266396e-10, -1.74203828e-01, -2.74171086e-09, -2.04233479e00, 1.00822319e-08, 6.07702672e-01, -1.14744314e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.63215449e-08, -1.55957475e-01, -4.13023287e-08, -2.02337551e00, 6.19276941e-09, 6.17800176e-01, -1.22737553e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.53878100e-08, -1.37765840e-01, -4.01130578e-08, -2.00424409e00, 8.78892692e-09, 6.27956867e-01, -9.04423425e-10, 3.99999991e-02, 3.99999991e-02],
        [-2.76206453e-08, -1.19621754e-01, -3.20486038e-08, -1.98493719e00, 1.20416024e-08, 6.38178647e-01, -5.03743880e-10, 3.99999991e-02, 3.99999991e-02],
        [3.28790861e-09, -1.01517305e-01, 2.02060590e-10, -1.96545100e00, 1.92992236e-08, 6.48471773e-01, -1.49719154e-10, 3.99999991e-02, 3.99999991e-02],
        [8.26621971e-09, -8.34447965e-02, 3.58132368e-09, -1.94578171e00, 2.07675335e-08, 6.58842325e-01, -6.48773257e-10, 3.99999991e-02, 3.99999991e-02],
        [-3.10538972e-08, -6.53969496e-02, -3.68909880e-08, -1.92592502e00, 1.69945942e-08, 6.69296741e-01, -9.79680670e-10, 3.99999991e-02, 3.99999991e-02],
        [-4.47211406e-08, -4.73656543e-02, -5.12603258e-08, -1.90587616e00, 1.58197118e-08, 6.79841638e-01, -9.02641240e-10, 3.99999991e-02, 3.99999991e-02],
        [-3.60921675e-08, -2.93423217e-02, -4.21610586e-08, -1.88562989e00, 1.78659629e-08, 6.90484107e-01, -7.59638796e-10, 3.99999991e-02, 3.99999991e-02],
        [-6.55031087e-08, -1.13188131e-02, -7.17394073e-08, -1.86517978e00, 1.41675525e-08, 7.01231301e-01, -9.09167130e-10, 3.99999991e-02, 3.99999991e-02],
        [-4.03416749e-08, 6.71289256e-03, -4.55443079e-08, -1.84452057e00, 1.79202448e-08, 7.12090373e-01, -1.60641822e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.67135672e-08, 2.47618891e-02, -6.15729476e-08, -1.82364511e00, 1.72999570e-08, 7.23069191e-01, -1.26030320e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.97667934e-08, 4.28366549e-02, -5.46052199e-08, -1.80254591e00, 2.12370068e-08, 7.34175861e-01, -1.77208115e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.52628111e-08, 6.09459244e-02, -7.86526613e-08, -1.78121614e00, 1.90223517e-08, 7.45418549e-01, -2.57672883e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.20635304e-08, 7.91001916e-02, -5.58252928e-08, -1.75964594e00, 2.54219970e-08, 7.56806672e-01, -2.63965383e-09, 3.99999991e-02, 3.99999991e-02],
        [3.11852233e-09, 9.73090604e-02, -3.83156973e-09, -1.73782647e00, 3.31004877e-08, 7.68349588e-01, -2.80139112e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.01767749e-08, 1.15582325e-01, -3.55535583e-08, -1.71574795e00, 3.01296481e-08, 7.80057132e-01, -3.19530402e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.18603701e-08, 1.33931100e-01, -7.42984128e-08, -1.69339919e00, 2.55036969e-08, 7.91940153e-01, -3.66310005e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.53870907e-08, 1.52367011e-01, -2.16636593e-08, -1.67076790e00, 3.36194361e-08, 8.04010332e-01, -3.57089314e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.59942886e-08, 1.70901135e-01, -2.20811387e-08, -1.64784181e00, 3.44677531e-08, 8.16279471e-01, -2.40884468e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.32197930e-08, 1.89546734e-01, -5.70507694e-08, -1.62460661e00, 3.35013368e-08, 8.28761160e-01, -2.51900589e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.81760038e-08, 2.08317742e-01, -3.45863995e-08, -1.60104585e00, 3.57773153e-08, 8.41470003e-01, -2.59438959e-09, 3.99999991e-02, 3.99999991e-02],
        [-8.68753176e-08, 2.27227420e-01, -8.71764385e-08, -1.57714415e00, 3.05043777e-08, 8.54420781e-01, -3.64601438e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.28496289e-08, 2.46291965e-01, -2.08034905e-08, -1.55288184e00, 4.29160671e-08, 8.67631078e-01, -3.38891470e-09, 3.99999991e-02, 3.99999991e-02],
        [8.66727135e-09, 2.65528083e-01, -1.50759405e-09, -1.52823877e00, 4.52586661e-08, 8.81119192e-01, -4.33550795e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.01861822e-08, 2.84953713e-01, -6.90719091e-08, -1.50319207e00, 4.10682972e-08, 8.94905567e-01, -2.13909268e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.65988856e-08, 3.04596156e-01, -5.60369884e-08, -1.47770560e00, 4.47941204e-08, 9.09018755e-01, -2.76203782e-10, 3.99999991e-02, 3.99999991e-02],
    ],
    dtype=np.float32,
)

FRANKA_LOW_TOOL_PATH_Q = np.array(
    [
        [-6.05379356e-08, 6.33147418e-01, -6.63244535e-08, -2.21152878e00, -8.81279760e-09, 7.82202244e-01, 6.36249009e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.76826881e-08, 6.39191270e-01, -2.94513480e-08, -2.19322920e00, -4.43570025e-09, 7.88876176e-01, 6.06098194e-09, 3.99999991e-02, 3.99999991e-02],
        [-9.61679802e-08, 6.45454824e-01, -1.05411843e-07, -2.17478013e00, -1.46922901e-08, 7.95659304e-01, 5.66232039e-09, 3.99999991e-02, 3.99999991e-02],
        [9.89441418e-09, 6.51933491e-01, 1.25805428e-08, -2.15618134e00, 1.78509207e-09, 8.02551448e-01, 4.91109642e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.62356802e-08, 6.58623576e-01, -4.98318826e-08, -2.13743091e00, -5.10390663e-09, 8.09553027e-01, 4.74702055e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.03264978e-08, 6.65523648e-01, -2.10067128e-08, -2.11852574e00, -1.77138926e-09, 8.16665411e-01, 4.03927869e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.64724322e-08, 6.72631323e-01, -3.83582019e-08, -2.09946322e00, -3.83497634e-09, 8.23889554e-01, 3.58274654e-09, 3.99999991e-02, 3.99999991e-02],
        [-1.09484368e-08, 6.79946840e-01, -1.18208181e-08, -2.08023810e00, 5.68838754e-10, 8.31227958e-01, 3.10846726e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.12576888e-08, 6.87467396e-01, -4.47804283e-08, -2.06084871e00, -4.02124289e-09, 8.38681757e-01, 2.19470464e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.29004158e-08, 6.95192873e-01, -2.49441996e-08, -2.04129004e00, -1.36708334e-09, 8.46253276e-01, 1.84675242e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.84489932e-08, 7.03123450e-01, -6.30119033e-08, -2.02155733e00, -5.23786703e-09, 8.53945017e-01, 1.58976965e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.71776795e-08, 7.11258888e-01, -6.22389251e-08, -2.00164628e00, -3.36265371e-09, 8.61759782e-01, 1.39518441e-09, 3.99999991e-02, 3.99999991e-02],
        [-2.12628581e-08, 7.19600320e-01, -2.46253098e-08, -1.98155105e00, 1.90067273e-09, 8.69700670e-01, 1.51307677e-09, 3.99999991e-02, 3.99999991e-02],
        [-5.45912933e-08, 7.28148103e-01, -5.93564025e-08, -1.96126628e00, -8.78570661e-10, 8.77770722e-01, 1.80158888e-10, 3.99999991e-02, 3.99999991e-02],
        [-6.16018312e-08, 7.36903727e-01, -6.65183606e-08, -1.94078612e00, -1.74196391e-09, 8.85973573e-01, 4.72132222e-10, 3.99999991e-02, 3.99999991e-02],
        [-3.65199284e-08, 7.45867848e-01, -4.04246414e-08, -1.92010522e00, -4.93440622e-10, 8.94312620e-01, -1.33998790e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.22370006e-08, 7.55044937e-01, -3.55669201e-08, -1.89921367e00, -4.17999191e-10, 9.02793527e-01, -2.33679986e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.20376543e-08, 7.64435887e-01, -3.63702171e-08, -1.87810659e00, 6.54652332e-10, 9.11419868e-01, -3.06667181e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.08276389e-08, 7.74043679e-01, -7.50592832e-08, -1.85677481e00, -2.47119303e-09, 9.20196950e-01, -4.74852957e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.72002704e-08, 7.83873558e-01, -5.15559790e-08, -1.83520901e00, 5.15734788e-10, 9.29130793e-01, -6.33247899e-09, 3.99999991e-02, 3.99999991e-02],
        [-4.33593037e-08, 7.93928444e-01, -4.72996504e-08, -1.81340039e00, 8.70716388e-10, 9.38226938e-01, -7.98056110e-09, 3.99999991e-02, 3.99999991e-02],
        [-7.37821182e-08, 8.04213285e-01, -7.67554056e-08, -1.79133904e00, -2.50420351e-09, 9.47492003e-01, -9.32366540e-09, 3.99999991e-02, 3.99999991e-02],
        [-6.26568379e-08, 8.14733505e-01, -6.63887079e-08, -1.76901376e00, -9.12253106e-10, 9.56933022e-01, -9.11676512e-09, 3.99999991e-02, 3.99999991e-02],
        [-3.44002835e-08, 8.25495958e-01, -3.85409464e-08, -1.74641132e00, 1.15198340e-09, 9.66558635e-01, -1.12907106e-08, 3.99999991e-02, 3.99999991e-02],
        [-1.55133293e-08, 8.36506367e-01, -2.04498249e-08, -1.72352076e00, 3.43680640e-09, 9.76376295e-01, -1.23565762e-08, 3.99999991e-02, 3.99999991e-02],
        [-3.66376440e-08, 8.47773075e-01, -4.08275262e-08, -1.70032656e00, 2.04073269e-09, 9.86395895e-01, -1.29591120e-08, 3.99999991e-02, 3.99999991e-02],
        [-5.41088880e-08, 8.59304070e-01, -5.74453196e-08, -1.67681444e00, 9.08515152e-10, 9.96627092e-01, -1.34987586e-08, 3.99999991e-02, 3.99999991e-02],
        [-8.06615645e-08, 8.71109247e-01, -8.16575607e-08, -1.65296721e00, 2.74467560e-10, 1.00708139e00, -1.30187718e-08, 3.99999991e-02, 3.99999991e-02],
        [-4.97791852e-08, 8.83199453e-01, -5.36381179e-08, -1.62876666e00, 2.43271669e-09, 1.01777077e00, -1.38752032e-08, 3.99999991e-02, 3.99999991e-02],
        [-1.53479363e-08, 8.95586371e-01, -2.24498606e-08, -1.60419285e00, 4.83429652e-09, 1.02870917e00, -1.58416000e-08, 3.99999991e-02, 3.99999991e-02],
        [-6.72103653e-08, 9.08283293e-01, -6.89928683e-08, -1.57922328e00, 1.02171538e-09, 1.03991127e00, -1.83358555e-08, 3.99999991e-02, 3.99999991e-02],
        [-6.07587864e-08, 9.21304762e-01, -6.29304679e-08, -1.55383503e00, 1.60169344e-09, 1.05139327e00, -1.86477411e-08, 3.99999991e-02, 3.99999991e-02],
        [2.94650349e-09, 9.34668183e-01, -6.04581629e-09, -1.52799964e00, 5.87273030e-09, 1.06317425e00, -2.20121308e-08, 3.99999991e-02, 3.99999991e-02],
    ],
    dtype=np.float32,
)

FRANKA_TOOL_PATH_Q_BY_MODE = {
    PUSH_MODE_HIGH: FRANKA_HIGH_TOOL_PATH_Q,
    PUSH_MODE_LOW: FRANKA_LOW_TOOL_PATH_Q,
    PUSH_MODE_CENTER: FRANKA_CENTER_TOOL_PATH_Q,
}

CAMERA_POS = wp.vec3(2.15, -1.20, 1.05)
CAMERA_PITCH = -17.0
CAMERA_YAW = 138.0
CAMERA_FOV = 50.0

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_NUM_FRAMES = FRAME_RATE * 4
VIDEO_NUM_FRAMES_BY_MODE = {
    PUSH_MODE_HIGH: FRAME_RATE * 4,
    PUSH_MODE_LOW: FRAME_RATE * 4,
    PUSH_MODE_CENTER: FRAME_RATE * 4,
}

FLOOR_COLOR = wp.vec3(0.52, 0.55, 0.57)
WALL_COLOR = wp.vec3(0.64, 0.43, 0.24)
TOOL_COLOR = wp.vec3(0.10, 0.15, 0.20)


@dataclass(frozen=True)
class SimulationResult:
    wall_body_index: int
    arm_body_index: int
    initial_wall_transform: np.ndarray
    final_wall_transform: np.ndarray
    initial_arm_transform: np.ndarray
    final_arm_transform: np.ndarray
    initial_body_poses: np.ndarray
    final_body_poses: np.ndarray
    max_wall_tilt_degrees: float
    final_wall_tilt_degrees: float
    initial_wall_top_position: np.ndarray
    final_wall_top_position: np.ndarray
    initial_tool_position: np.ndarray
    final_tool_position: np.ndarray
    tool_positions: np.ndarray
    max_tool_path_x_deviation: float
    max_tool_path_z_deviation: float
    final_wall_velocity_norm: float


def _quat_z(degrees: float) -> wp.quat:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(degrees))


def _normalize_friction_mode(push_mode: str, friction_mode: str | None) -> str:
    push_mode = _validate_push_mode(push_mode)
    friction_mode = DEFAULT_FRICTION_MODE if friction_mode is None else friction_mode
    if push_mode in {PUSH_MODE_HIGH, PUSH_MODE_LOW}:
        if friction_mode != DEFAULT_FRICTION_MODE:
            raise ValueError("high and low contact-height pushes use the fixed default wall friction")
        return friction_mode
    if friction_mode in FRICTION_COEFFICIENT_BY_MODE:
        return friction_mode
    modes = ", ".join(sorted(FRICTION_COEFFICIENT_BY_MODE))
    raise ValueError(f"center push requires friction_mode to be one of: {modes}")


def _friction_coefficient(material: MaterialPreset, friction_mode: str) -> float:
    if friction_mode == DEFAULT_FRICTION_MODE:
        return material.friction
    return FRICTION_COEFFICIENT_BY_MODE[friction_mode]


def _wall_shape_cfg(
    material: MaterialPreset,
    friction_mode: str = DEFAULT_FRICTION_MODE,
) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=material.density * WALL_DENSITY_SCALE,
        mu=_friction_coefficient(material, friction_mode),
        restitution=min(material.restitution, 0.02),
    )


def _ground_shape_cfg(
    material: MaterialPreset,
    friction_mode: str = DEFAULT_FRICTION_MODE,
) -> newton.ModelBuilder.ShapeConfig:
    cfg = _static_shape_cfg(material)
    cfg.mu = _friction_coefficient(material, friction_mode)
    cfg.restitution = min(material.restitution, 0.02)
    return cfg


def _tool_shape_cfg(material: MaterialPreset) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=material.density,
        mu=max(0.45, material.friction),
        restitution=min(material.restitution, 0.02),
    )


def _find_body(builder: newton.ModelBuilder, suffix: str) -> int:
    return next(i for i, label in enumerate(builder.body_label) if label.endswith(f"/{suffix}"))


def _quat_to_matrix_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wall_tilt_degrees(wall_transform: np.ndarray) -> float:
    rotation = _quat_to_matrix_xyzw(wall_transform[3:7])
    wall_up = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return math.degrees(math.acos(float(np.clip(wall_up[2], -1.0, 1.0))))


def wall_top_position(wall_transform: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix_xyzw(wall_transform[3:7])
    return wall_transform[:3] + rotation @ np.array([0.0, 0.0, WALL_HEIGHT * 0.5], dtype=np.float64)


def _transform_point(transform_xyzw: np.ndarray, point: wp.vec3) -> np.ndarray:
    rotation = _quat_to_matrix_xyzw(transform_xyzw[3:7])
    return transform_xyzw[:3] + rotation @ np.array([float(point[0]), float(point[1]), float(point[2])])


def _validate_push_mode(push_mode: str) -> str:
    if push_mode not in FRANKA_TOOL_PATH_Q_BY_MODE:
        modes = ", ".join(sorted(FRANKA_TOOL_PATH_Q_BY_MODE))
        raise ValueError(f"unknown push_mode {push_mode!r}; expected one of: {modes}")
    return push_mode


def _tool_path_progress(time_seconds: float, push_mode: str) -> tuple[float, float]:
    _validate_push_mode(push_mode)
    accel_seconds = ROBOT_ACCEL_SECONDS
    if time_seconds <= ROBOT_HOLD_SECONDS:
        return 0.0, 0.0
    if time_seconds >= ROBOT_HOLD_SECONDS + accel_seconds:
        return 1.0, 0.0

    normalized = (time_seconds - ROBOT_HOLD_SECONDS) / accel_seconds
    return normalized * normalized, 2.0 * normalized / accel_seconds


def franka_joint_target(time_seconds: float, push_mode: str = DEFAULT_PUSH_MODE) -> tuple[np.ndarray, np.ndarray]:
    push_mode = _validate_push_mode(push_mode)
    tool_path_q = FRANKA_TOOL_PATH_Q_BY_MODE[push_mode]
    progress, progress_rate = _tool_path_progress(time_seconds, push_mode)
    sample = progress * (len(tool_path_q) - 1)
    index = min(int(math.floor(sample)), len(tool_path_q) - 2)
    local = sample - index
    q0 = tool_path_q[index]
    q1 = tool_path_q[index + 1]
    q = q0 * (1.0 - local) + q1 * local
    qd = (q1 - q0) * (len(tool_path_q) - 1) * progress_rate
    return q.astype(np.float32), qd.astype(np.float32)


def franka_tool_target(time_seconds: float, push_mode: str = DEFAULT_PUSH_MODE) -> tuple[np.ndarray, np.ndarray]:
    push_mode = _validate_push_mode(push_mode)
    progress, progress_rate = _tool_path_progress(time_seconds, push_mode)
    path_length = TOOL_PATH_END_Y - TOOL_PATH_START_Y
    tool_path_z = TOOL_PATH_Z_BY_MODE[push_mode]
    return (
        np.array([TOOL_PATH_X, TOOL_PATH_START_Y + path_length * progress, tool_path_z], dtype=np.float32),
        np.array([0.0, path_length * progress_rate, 0.0], dtype=np.float32),
    )


class RoboarmWallDemo:
    def __init__(
        self,
        viewer: Any,
        arm_material: str = "steel",
        wall_material: str = "wood",
        push_mode: str = DEFAULT_PUSH_MODE,
        friction_mode: str | None = None,
    ):
        self.viewer = viewer
        self.push_mode = _validate_push_mode(push_mode)
        self.friction_mode = _normalize_friction_mode(self.push_mode, friction_mode)
        self.tool_path_z = TOOL_PATH_Z_BY_MODE[self.push_mode]
        self.fps = FRAME_RATE
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        arm_preset = MATERIALS[arm_material]
        wall_preset = MATERIALS[wall_material]
        tool_cfg = _tool_shape_cfg(arm_preset)
        wall_cfg = _wall_shape_cfg(wall_preset, self.friction_mode)
        ground_cfg = _ground_shape_cfg(wall_preset, self.friction_mode)

        builder = newton.ModelBuilder()
        builder.rigid_gap = RIGID_GAP

        asset_root = newton.utils.download_asset(FRANKA_ASSET_NAME)
        builder.add_urdf(
            asset_root / FRANKA_URDF,
            xform=wp.transform(p=FRANKA_BASE_POS, q=_quat_z(FRANKA_BASE_YAW_DEG)),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        self.robot_body_count = builder.body_count
        robot_shape_count = builder.shape_count
        self.hand_body_index = _find_body(builder, FRANKA_HAND_LABEL)
        self.arm_body_index = _find_body(builder, FRANKA_TCP_LABEL)

        for body_index in range(self.robot_body_count):
            builder.body_flags[body_index] = int(newton.BodyFlags.KINEMATIC)
        for shape_index in range(robot_shape_count):
            builder.shape_flags[shape_index] = int(builder.shape_flags[shape_index]) & ~int(
                newton.ShapeFlags.COLLIDE_SHAPES
            )

        builder.joint_q[:FRANKA_DOF_COUNT] = FRANKA_TOOL_PATH_Q_BY_MODE[self.push_mode][0].tolist()
        builder.add_shape_sphere(
            body=self.hand_body_index,
            xform=wp.transform(p=TOOL_LOCAL_POS, q=wp.quat_identity()),
            radius=TOOL_RADIUS,
            cfg=tool_cfg,
            color=TOOL_COLOR,
            label="franka_push_tool",
        )

        floor_center = wp.vec3(0.0, FLOOR_CENTER_Y, -FLOOR_THICKNESS * 0.5)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=floor_center, q=wp.quat_identity()),
            hx=FLOOR_WIDTH * 0.5,
            hy=FLOOR_LENGTH * 0.5,
            hz=FLOOR_THICKNESS * 0.5,
            cfg=ground_cfg,
            color=FLOOR_COLOR,
            label="floor",
        )

        wall_center = wp.vec3(0.0, WALL_CENTER_Y, WALL_HEIGHT * 0.5)
        self.wall_body_index = builder.add_body(
            xform=wp.transform(p=wall_center, q=wp.quat_identity()),
            label="wall",
        )
        self.wall_joint_index = builder.joint_count - 1
        builder.add_shape_box(
            body=self.wall_body_index,
            hx=WALL_WIDTH * 0.5,
            hy=WALL_THICKNESS * 0.5,
            hz=WALL_HEIGHT * 0.5,
            cfg=wall_cfg,
            color=WALL_COLOR,
            label="wall_shape",
        )

        self.model = builder.finalize()
        self.wall_q_start = int(self.model.joint_q_start.numpy()[self.wall_joint_index])
        self.wall_qd_start = int(self.model.joint_qd_start.numpy()[self.wall_joint_index])
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=SOLVER_ITERATIONS,
            rigid_contact_relaxation=SOLVER_CONTACT_RELAXATION,
            angular_damping=SOLVER_ANGULAR_DAMPING,
            enable_restitution=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=CAMERA_POS, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV

        self.initial_body_poses = self.body_poses()
        self.initial_tool_position = self.tool_position()
        self.tool_position_history = [self.initial_tool_position.copy()]
        self.max_wall_tilt_degrees = self.wall_tilt_degrees()

    def body_poses(self) -> np.ndarray:
        return self.state_0.body_q.numpy().copy()

    def wall_tilt_degrees(self) -> float:
        return wall_tilt_degrees(self.state_0.body_q.numpy()[self.wall_body_index])

    def tool_position(self) -> np.ndarray:
        return _transform_point(self.state_0.body_q.numpy()[self.hand_body_index], TOOL_LOCAL_POS)

    def _set_robot_pose(self, state: Any, time_seconds: float) -> None:
        q_target, qd_target = franka_joint_target(time_seconds, self.push_mode)
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()

        # The wall is a free body. Keep its generalized state synced before FK
        # updates the kinematic robot, otherwise eval_fk would reset the wall.
        joint_q[self.wall_q_start : self.wall_q_start + 7] = body_q[self.wall_body_index]
        joint_qd[self.wall_qd_start : self.wall_qd_start + 6] = body_qd[self.wall_body_index]
        joint_q[:FRANKA_DOF_COUNT] = q_target
        joint_qd[:FRANKA_DOF_COUNT] = qd_target
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def simulate(self) -> None:
        for substep_index in range(self.sim_substeps):
            self._set_robot_pose(self.state_0, self.sim_time + substep_index * self.sim_dt)
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        self.max_wall_tilt_degrees = max(self.max_wall_tilt_degrees, self.wall_tilt_degrees())
        self.tool_position_history.append(self.tool_position())

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def result(self) -> SimulationResult:
        final_body_poses = self.body_poses()
        final_body_velocities = self.state_0.body_qd.numpy().copy()
        initial_wall = self.initial_body_poses[self.wall_body_index].copy()
        final_wall = final_body_poses[self.wall_body_index].copy()
        tool_positions = np.asarray(self.tool_position_history, dtype=np.float64)
        return SimulationResult(
            wall_body_index=self.wall_body_index,
            arm_body_index=self.arm_body_index,
            initial_wall_transform=initial_wall,
            final_wall_transform=final_wall,
            initial_arm_transform=self.initial_body_poses[self.arm_body_index].copy(),
            final_arm_transform=final_body_poses[self.arm_body_index].copy(),
            initial_body_poses=self.initial_body_poses.copy(),
            final_body_poses=final_body_poses,
            max_wall_tilt_degrees=float(self.max_wall_tilt_degrees),
            final_wall_tilt_degrees=float(wall_tilt_degrees(final_wall)),
            initial_wall_top_position=wall_top_position(initial_wall),
            final_wall_top_position=wall_top_position(final_wall),
            initial_tool_position=self.initial_tool_position.copy(),
            final_tool_position=tool_positions[-1].copy(),
            tool_positions=tool_positions.copy(),
            max_tool_path_x_deviation=float(np.max(np.abs(tool_positions[:, 0] - TOOL_PATH_X))),
            max_tool_path_z_deviation=float(np.max(np.abs(tool_positions[:, 2] - self.tool_path_z))),
            final_wall_velocity_norm=float(np.linalg.norm(final_body_velocities[self.wall_body_index])),
        )


def build_scene(
    *,
    arm_material: str = "steel",
    wall_material: str = "wood",
    push_mode: str = DEFAULT_PUSH_MODE,
    friction_mode: str | None = None,
    viewer: str | Any = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> RoboarmWallDemo:
    if device:
        wp.set_device(device)

    if isinstance(viewer, str):
        viewer_obj = _build_viewer(viewer, num_frames=num_frames, output_path=output_path)
    else:
        viewer_obj = viewer

    return RoboarmWallDemo(
        viewer=viewer_obj,
        arm_material=arm_material,
        wall_material=wall_material,
        push_mode=push_mode,
        friction_mode=friction_mode,
    )


def run_simulation(
    *,
    arm_material: str = "steel",
    wall_material: str = "wood",
    push_mode: str = DEFAULT_PUSH_MODE,
    friction_mode: str | None = None,
    viewer: str = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> SimulationResult:
    demo = build_scene(
        arm_material=arm_material,
        wall_material=wall_material,
        push_mode=push_mode,
        friction_mode=friction_mode,
        viewer=viewer,
        num_frames=num_frames,
        output_path=output_path,
        device=device,
    )

    try:
        while demo.viewer.is_running():
            if not demo.viewer.is_paused():
                demo.step()
            demo.render()
        return demo.result()
    finally:
        demo.viewer.close()


def default_video_output_path(push_mode: str = DEFAULT_PUSH_MODE, friction_mode: str | None = None) -> str:
    push_mode = _validate_push_mode(push_mode)
    friction_mode = _normalize_friction_mode(push_mode, friction_mode)
    if push_mode == PUSH_MODE_CENTER:
        return f"outputs/roboarm_wall_center_{friction_mode}_friction.mp4"
    return {
        PUSH_MODE_HIGH: "outputs/roboarm_wall_high.mp4",
        PUSH_MODE_LOW: "outputs/roboarm_wall_low.mp4",
    }[push_mode]


def render_video(
    *,
    output_path: str | None = None,
    arm_material: str = "steel",
    wall_material: str = "wood",
    push_mode: str = DEFAULT_PUSH_MODE,
    friction_mode: str | None = None,
    num_frames: int = VIDEO_NUM_FRAMES,
    device: str | None = None,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render video output")

    output = Path(output_path or default_video_output_path(push_mode, friction_mode))
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    viewer = newton.viewer.ViewerGL(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, headless=True)
    demo = build_scene(
        arm_material=arm_material,
        wall_material=wall_material,
        push_mode=push_mode,
        friction_mode=friction_mode,
        viewer=viewer,
        num_frames=num_frames,
        output_path=None,
        device=device,
    )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r",
        str(FRAME_RATE),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        for frame_index in range(num_frames):
            if frame_index > 0:
                demo.step()
            demo.render()
            frame = viewer.get_frame()
            frame_np = np.ascontiguousarray(frame.numpy())
            assert process.stdin is not None
            process.stdin.write(frame_np.tobytes())

        assert process.stdin is not None
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")
        return output
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()
        viewer.close()
