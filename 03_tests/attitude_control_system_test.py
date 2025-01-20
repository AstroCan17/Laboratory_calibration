import sys
sys.path.append('D:/03_cdk_processing/07_hyperspectral_lab_cal/Laboratory_calibration/02_scripts')

import unittest
import numpy as np
from attitude_control_system import AOCSController

class TestAOCSController(unittest.TestCase):
    
    def setUp(self):
        self.controller = AOCSController(roll_angle=30, pitch_angle=45, yaw_angle=60)
    
    def test_euler_to_matrix_ned(self):
        matrix = self.controller.euler_to_matrix_ned()
        expected_matrix = np.array([
            [0.35355339, -0.5732233, 0.73919892],
            [0.61237244, 0.73919892, 0.28033009],
            [-0.70710678, 0.35355339, 0.61237244]
        ])
        np.testing.assert_almost_equal(matrix, expected_matrix, decimal=5)

    def test_euler_to_matrix_ned_zero_angles(self):
        controller = AOCSController(roll_angle=0, pitch_angle=0, yaw_angle=0)
        matrix = controller.euler_to_matrix_ned()
        expected_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_almost_equal(matrix, expected_matrix, decimal=5)

if __name__ == '__main__':
    unittest.main()