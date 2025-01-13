import os
from os.path import dirname, basename
import sys
sys.path.append('D:/03_cdk_processing/07_hyperspectral_lab_cal/Laboratory_calibration/02_scripts')

import matplotlib.pyplot as plt

from mathematical_modelling import TheorySec2
import logging 

# Set up logging
logging.basicConfig(level=logging.INFO)
__name__ = "run_model"
LOG = logging.getLogger(__name__)

#initializing the class
theory = TheorySec2(
    altitude_m=500e3,          # Altitude [m]
    t_int=400,                 # Integration time [s]
    yaw_angle_deg=0,           # Yaw angle in degrees (rotation about the z-axis)
    roll_angle_deg=0,          # Roll angle in degrees (rotation about the x-axis)
    pitch_angle_deg=0,         # Pitch angle in degrees (rotation about the y-axis)
    temperature=20  
)



# Log the class attributes
LOG.info(f"Altitude: {int(theory.altitude_m*1e-3)} km")
LOG.info(f"Integration time: {theory.t_int} s")
LOG.info(f"Yaw angle: {theory.yaw_angle} degrees")
LOG.info(f"Roll angle: {theory.roll_angle} degrees")
LOG.info(f"Pitch angle: {theory.pitch_angle} degrees")
LOG.info(f"Temperature: {theory.T} °C")







