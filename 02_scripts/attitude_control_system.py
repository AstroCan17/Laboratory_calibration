

import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
__name__ = "AOCS Controller"
LOG = logging.getLogger(__name__)



class AOCSController:
    def __init__(self, roll_angle, pitch_angle, yaw_angle):
        self.roll_angle = roll_angle
        self.pitch_angle = pitch_angle
        self.yaw_angle = yaw_angle

    def euler_to_matrix(self, degrees=True):
        """
        Converts roll, pitch, and yaw angles to a rotation matrix.
        Rotation order: Rz(yaw) * Ry(pitch) * Rx(roll) (intrinsic rotations)
        
        If degrees=True, angles are assumed to be in degrees and are converted to radians.
        Returns a 3x3 rotation matrix.
        """
        if degrees:
            roll = np.deg2rad(self.roll_angle)
            pitch = np.deg2rad(self.pitch_angle)
            yaw = np.deg2rad(self.yaw_angle)

        # Rotation matrix for roll
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        # Rotation matrix for pitch
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        # Rotation matrix for yaw
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        # Combined rotation matrix (Rz * Ry * Rx)
        R = np.dot(np.dot(Rz, Ry), Rx)
        return R

    def compute_alpha_beta(self, degrees=True):
        """
        Uses given roll, pitch, and yaw angles to compute the direction of the sensor boresight
        (assumed to be in the direction of the z' axis) in the world/aircraft coordinates.
        Then calculates the alpha (along-track) and beta (across-track) angles based on thesis equations (B.4) and (B.5).
        
        Input:
        roll, pitch, yaw (in degrees or radians, depending on the 'degrees' parameter)
        Output:
        alpha, beta (returned in degrees, can be used in radians if required)
        """
        # Generate the rotation matrix
        R = self.euler_to_matrix(degrees=degrees)

        # Assume sensor boresight is in the direction of the z' axis (down)
        v_s = np.array([0.0, 0.0, 1.0])

        # Vector transformed to world/aircraft coordinates
        v_e = np.dot(R, v_s)  # [x', y', z']^T

        # Extract components
        x_p, y_p, z_p = v_e

        # Calculate alpha (along-track) using arctan2(z', x')
        alpha_rad = np.arctan2(z_p, x_p)

        # Calculate beta (across-track) using arctan2(z', y')
        beta_rad = np.arctan2(z_p, y_p)

        if degrees:
            alpha = np.rad2deg(alpha_rad)
            beta = np.rad2deg(beta_rad)
            LOG.info(("-" * 50) + "\n"+
                     f"Along Track Alpha: {alpha:.2f} degrees\n"
                     f"Across Track Beta: {beta:.2f} degrees\n"
                     + ("-" * 50))
            
            return alpha, beta
        else:
            LOG.info(("-" * 50) + "\n"+
                     f"Along Track Alpha: {alpha:.2f} radians\n"
                     f"Across Track Beta: {beta:.2f} radians\n"
                     +("-" * 50))
            return alpha_rad, beta_rad

