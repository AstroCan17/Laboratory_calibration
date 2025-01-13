from Py6S import *
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def calculate_total_irradiance(altitude, lambda_min, lambda_max, yaw_angle, roll_angle, pitch_angle):
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = 0  # Solar zenith angle
    s.geometry.solar_a = 0  # Solar azimuth angle

    # Convert angles to radians
    yaw_rad = np.radians(yaw_angle)
    roll_rad = np.radians(roll_angle)
    pitch_rad = np.radians(pitch_angle)

    # Rotation matrices
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    # Initial view vector (pointing along the x-axis)
    v = np.array([0.0, 0.0, 1.0])

    # Apply rotations: first roll, then pitch, then yaw
    v_rot = Rz @ (Ry @ (Rx @ v))

    # Extract the rotated vector components
    x_p, y_p, z_p = v_rot

    # Calculate the new view zenith and azimuth angles
    view_zenith = np.degrees(np.arccos(z_p))
    view_azimuth = np.degrees(np.arctan2(y_p, x_p))

    # Set the view angles in SixS
    s.geometry.view_z = view_zenith  # View zenith angle
    s.geometry.view_a = view_azimuth  # View azimuth angle

    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(altitude / 1000)  # Convert to km
    lambda_min = lambda_min * 1e-3  # Convert to um
    lambda_max = lambda_max * 1e-3  # Convert to um

    total_irradiance = 0

    for wavelength in range(int(lambda_min), int(lambda_max) + 1):
        s.wavelength = Wavelength(wavelength)
        s.run()
        total_irradiance += s.outputs.apparent_radiance
    L_lambda = total_irradiance * 1e-3  # Convert W/m²/sr/um to W/m²/sr/nm
    LOG.info(f'Total irradiance: {L_lambda:.2f} W/m²/sr/nm\n'
             f'View zenith angle: {view_zenith:.2f} degrees\n'
             f'View azimuth angle: {view_azimuth:.2f} degrees')
    return L_lambda