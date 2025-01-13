import os
from os.path import dirname, basename
import sys
sys.path.append('D:/03_cdk_processing/07_hyperspectral_lab_cal/Laboratory_calibration/02_scripts')

import matplotlib.pyplot as plt

from mathematical_modelling import TheorySec2

#initializing the class
theory = TheorySec2(
    altitude_m=500e3,          # Altitude [m]
    off_nadir_deg=0,           # Off-nadir angle in the along-track direction [degrees]
    t_int=400,                 # Integration time [s]
    yaw_angle_deg=0,           # Yaw angle in degrees (rotation about the z-axis)
    roll_angle_deg=0,          # Roll angle in degrees (rotation about the x-axis)
    pitch_angle_deg=0,         # Pitch angle in degrees (rotation about the y-axis)
    temperature=20  
)





