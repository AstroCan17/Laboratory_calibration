import numpy as np
import radianceAtSensor as ras
import math
import pandas as pd
import os 
import interpolate_qe as iq
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG)

LOG =logging.getLogger("TheorySec2")    


class TheorySec2:
    
    def __init__(
        self,
        altitude_m=int,           # Satellite altitude [m] (e.g. 705 km)
        t_int=float,              # Integration time [s]
        yaw_angle_deg = float,    # Yaw angle in degrees (rotation about the z-axis)
        roll_angle_deg = float,   # Roll angle in degrees (rotation about the x-axis)
        pitch_angle_deg = float,  # Pitch angle in degrees (rotation about the y-axis)
        temperature = int,         # Detector temperature [°C]
        lambda_ = float           # Center wavelength for methane detection [nm]

    ):

        # Physical / orbital constants
        self.h_planck = 6.63e-34      # Planck's constant [J*s]
        self.c_light  = 3.0e8         # Speed of light [m/s]
        self.G        = 6.67e-11      # Gravitational constant [m^3 / kg / s^2]
        self.M_earth  = 5.97e24       # Earth mass [kg]
        self.mu       = 3.986004418e14  # GM, [m^3 / s^2]
        
        # Earth radius [m] (from the referenced paper or typical Earth radius)
        self.R_earth = 6731e3         # Mean Earth radius [m]     
        
        # Satellite/orbit params
        self.altitude_m    = float(altitude_m)  # Satellite altitude [m]

        
        # Radiometric params
             
        self.OE          = 0.6          # Optical efficiency [dimensionless]   worst case scenario
        self.T           = temperature  # Detector temperature [°C]
        self.lambda_min  = float(1580)  # [nm] Minimum wavelength
        self.lambda_max  = float(1690)  # [nm] Maximum wavelength
        self.delta_lambda_nm = (self.lambda_max - self.lambda_min)
        # self.QE          = 0.18         # Quantum efficiency [dimensionless] for lambda_c_nm = 1666.2 nm
        self.t_int       = t_int                # Integration time [s]
        
        # Noise params
        self.dark_current_rate = 600      # [e- / pix / s]
        self.read_noise_e      = 30       # [e- rms / pix]
        
        # Optic/detector
        self.f_m         = 0.18               # [m] focal length  
        self.aperture_d = float(0.072)        # m     
        self.aperture_r = self.aperture_d / 2 # m    
        self.pixel_pitch  = 15e-6              # [m]

        self.ifov =  0.15                     # corresponding to 100 meters across track

        self.fov_act = 20                     #  km across track

            # Detector dimensions (640 x 512), but 2 pixels are dead
        self.nx          = 638    # pixel count (across-track)
        self.ny          = 510     # pixel count (along-track)

        self.w_y     = self.pixel_pitch*self.ny  # [m] along-track dimension
        self.w_z     = self.pixel_pitch*self.nx  # [m] across-track dimension
        self.delta_y = self.pixel_pitch         # [m] along-track pixel size
        self.delta_z = self.pixel_pitch         # [m] across-track pixel size
        
        # "center" wavelength in nm for methane detection
        self.lambda_ = lambda_           # nm (e.g. for methane detection)

        # Optics Forward model parameters and functions

        # self.o_optical() = o_optical_fwm
        self.yaw_angle = yaw_angle_deg
        self.roll_angle = roll_angle_deg
        self.pitch_angle = pitch_angle_deg

        self.L_lambda = ras.calculate_total_irradiance(self.altitude_m, self.lambda_min, self.lambda_max, self.yaw_angle, self.roll_angle, self.pitch_angle)

        # directories
        self.spectral_responsivity_path = "D:/03_cdk_processing/07_hyperspectral_lab_cal/Laboratory_calibration/00_data/01_quantum_efficiency"
        interpolated_file_name =  f'interpolated_data_{int(self.T)}.csv'
        if not os.path.exists(self.spectral_responsivity_path+"/"+ interpolated_file_name):
            LOG.info(f"Interpolated Quantum Efficieny does not exist at {self.spectral_responsivity_path}")
            self.interpolated_QE = iq.run_interpolation(self.T, self.spectral_responsivity_path)
            self.wavelengths_common = self.interpolated_QE['Wavelength']
            self.efficiencies_target = self.interpolated_QE['Efficiency']
        else:
            path = self.spectral_responsivity_path+"/"+ interpolated_file_name
            LOG.info(f"Interpolated Quantum Efficieny already exists at {path}")
            self.interpolated_QE = pd.read_csv(path)
            self.wavelengths_common = self.interpolated_QE['Wavelength']
            self.efficiencies_target = self.interpolated_QE['Efficiency']


        # Approximate the quantum efficiency of the given wavelength
        qe_interp = interp1d(self.wavelengths_common, self.efficiencies_target, kind='cubic', fill_value="extrapolate")
        self.qe_lambda = qe_interp(self.lambda_)
        self.qe_lambda = self.qe_lambda/100 # convert to percentage
        LOG.info(f"Quantum efficiency at {self.lambda_} nm: % {self.qe_lambda:.2f}")


        plt.figure()
        plt.plot(self.wavelengths_common, self.efficiencies_target, label='Interpolated QE')
        plt.scatter(self.lambda_, self.qe_lambda, color='red', label=f'Target wavelength: {self.lambda_} nm')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Quantum Efficiency [%]')
        plt.title(f'Interpolated QE at {self.T} °C')
        plt.legend()
        plt.grid()
        plt.show()





    # def compute_alpha_beta_from_yaw_roll(self):
    #     """
    #     Computes the along-track (alpha) and across-track (beta) angles in degrees 
    #     from a satellite's line of sight (LoS), given yaw and roll angles (in degrees).

    #     In this context:
    #     - alpha (along-track) corresponds to how much the LoS deviates 
    #         in the forward/backward (x-axis) direction of the satellite.
    #     - beta (across-track) corresponds to the lateral (side-to-side) deviation 
    #         in the satellite's y–z plane.

    #     Parameters
    #     ----------
    #     yaw_deg : float
    #         Yaw angle in degrees (rotation about the z-axis).
    #     roll_deg : float
    #         Roll angle in degrees (rotation about the x-axis).

    #     Returns
    #     -------
    #     alpha_deg : float
    #         Along-track angle in degrees.
    #     beta_deg : float
    #         Across-track angle in degrees.

    #     Notes
    #     -----
    #     This function assumes pitch = 0 (i.e., no rotation about the y-axis). 
    #     If you need to account for pitch, you can include an additional rotation matrix 
    #     R_y(pitch) in the multiplication sequence and then compute alpha and beta 
    #     in a similar manner.

    #     Appendix B Reference:
    #     ---------------------
    #     In Appendix B of the thesis, a spherical coordinate system is introduced, 
    #     where the angles alpha and beta are defined as follows:
        
    #     - beta is equivalent to the azimuth angle φ (across-track).
    #     - alpha is the along-track angle, defined by the relationship 
    #         tan(alpha) = (z') / (x'), or equivalently alpha = arctan2(z', x').
        
    #     There, x', y', and z' are the Cartesian coordinates obtained after applying 
    #     the relevant rotation matrices (from roll, pitch, yaw) to the initial 
    #     body-fixed coordinate system aligned with the satellite's nominal flight direction.
        
    #     For more details on the underlying mathematics, see Appendix B.1–B.9 of the thesis,
    #     where the following are presented:
    #     - (B.1)–(B.3): Cartesian-to-spherical coordinate transformations.
    #     - (B.4)–(B.8): Definitions of α (along-track) and β (across-track).
    #     - (B.9) : Additional trigonometric relationships used for α and β.
    #     """
    #     # 1) Dereceden radyana çevir
    #     yaw_rad  = np.deg2rad(self.yaw_angle)
    #     roll_rad = np.deg2rad(self.roll_angle)
        
    #     # 2) Roll matrisi R_x(roll)
    #     #    roll pozitif ise x-ekseni etrafında saat yönünde (uçaktan bakınca)
    #     Rx = np.array([
    #         [1,            0,             0           ],
    #         [0,  np.cos(roll_rad), -np.sin(roll_rad) ],
    #         [0,  np.sin(roll_rad),  np.cos(roll_rad) ]
    #     ])
        
    #     # 3) Yaw matrisi R_z(yaw)
    #     #    yaw pozitif ise z-ekseni etrafında "sola dönme" kabulü
    #     Rz = np.array([
    #         [ np.cos(yaw_rad), -np.sin(yaw_rad),  0 ],
    #         [ np.sin(yaw_rad),  np.cos(yaw_rad),  0 ],
    #         [               0,                0,  1 ]
    #     ])
        
    #     # 4) Başlangıç bakış vektörü (x-ekseni yönünde)
    #     v = np.array([1.0, 0.0, 0.0])
        
    #     # 5) Dönüşleri uygula
    #     #    "Önce roll, sonra yaw" sıralaması:
    #     v_rot = Rz @ (Rx @ v)
        
    #     # 6) Dönen vektörden (x', y', z') değerlerini al
    #     x_p, y_p, z_p = v_rot
        
    #     # 7) alpha ve beta hesapla
    #     #    beta = arctan2(z', y')
    #     beta_rad  = math.atan2(z_p, y_p)
        
    #     #    alpha = arctan2(z', x')
    #     alpha_rad = math.atan2(z_p, x_p)
        
    #     # 8) Radyandan dereceye çevir
    #     alpha_deg = np.rad2deg(alpha_rad)
    #     beta_deg  = np.rad2deg(beta_rad)
        
    #     return alpha_deg, beta_deg



    def optical_system_function(self):
        # Optical system function that describes beam shaping and transmission
        return self.OE  # Simplified example using optical efficiency
    


    def calculate_irradiance_simplified(self):
        """
        Calculate the irradiance E(y, z) based on the simplified equation (2.2).

        Parameters:
        optical_system_function (function): Optical system function O(y, z, alpha, beta, lambda) [unitless]
        L_lambda (function): Spectral radiance field L_lambda(alpha, beta, lambda) [W m^-2 nm^-1 sr^-1]
        A_e (float): Entrance aperture area [m^2]
        alpha_range (tuple): Range of alpha values (min, max) [radians]
        beta_range (tuple): Range of beta values (min, max) [radians]
        lambda_range (tuple): Range of lambda values (min, max) [nm]

        Returns:
        float: Calculated irradiance E(y, z) [W m^-2]
        """
        # alpha_values  = np.linspace(alpha_range[0], alpha_range[1], 100)
        # beta_values   = np.linspace(beta_range[0], beta_range[1], 100)
        # lambda_values = np.linspace(lambda_range[0], lambda_range[1], 100)

        # E_yz = 0
        # for alpha in alpha_values:
        #     for beta in beta_values:
        #         for lambda_ in lambda_values:
        #             E_yz += O(alpha, beta, lambda_) * L_lambda(alpha, beta, lambda_)

        # E_yz *= A_e
        E_yz = self.optical_system_function()* self.L_lambda

        return E_yz




    def calculate_photo_current(self,O, D_i, A_e, A_d, T):
        """
        Calculate the photo current I_i^ph(T) based on equation (2.3).

        Parameters:
        O (function): Optical system function O(y, z, alpha, beta, lambda) [unitless]
        L_lambda (function): Spectral radiance field L_lambda(alpha, beta, lambda) [W m^-2 nm^-1 sr^-1]
        D_i (function): FPA model D_i(lambda, T, y, z) [e^- / photons]
        A_e (float): Entrance aperture area [m^2]
        A_d (float): Detector area [m^2]
        alpha_range (tuple): Range of alpha values (min, max) [radians]
        beta_range (tuple): Range of beta values (min, max) [radians]
        lambda_range (tuple): Range of lambda values (min, max) [nm]
        T (float): Detector temperature [K]
        w_y (float): Width of the pixel's photosensitive area along the y-axis [m]
        w_z (float): Width of the pixel's photosensitive area along the z-axis [m]
        delta_y (float): Pixel pitch along the y-axis [m]
        delta_z (float): Pixel pitch along the z-axis [m]

        Returns:
        float: Calculated photo current I_i^ph(T) [e^- / s]
        """
        alpha_angle , beta_angle = self.compute_alpha_beta()
        lambda_range = (self.lambda_min, self.lambda_max)
        alpha_values  = np.linspace(alpha_angle, alpha_angle[1], 100)
        beta_values   = np.linspace(beta_angle[0], beta_angle[1], 100)
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], 100)

        I_ph = 0
        for alpha in alpha_values:
            for beta in beta_values:
                for lambda_ in lambda_values:
                    for y in np.linspace(-self.w_y/2, self.w_y/2, 10):
                        for z in np.linspace(-self.w_z/2, self.w_z/2, 10):
                            I_ph += D_i(lambda_, T, y, z, 0, self.w_y, self.w_z, self.delta_y, self.delta_z) * O(alpha, beta, lambda_) * self.L_lambda

        I_ph *= A_e * A_d
        return I_ph


    def sqcap(self,w, x):
        """
        Boxcar function sqcap_w(x) based on equation (2.5).

        Parameters:
        w (float): Width of the boxcar function
        x (float): Input value

        Returns:
        int: 1 if |x| > w/2, 0 otherwise
        """
        return 1 if abs(x) > w / 2 else 0


    def D_i(self):
        """
        FPA model D_i(lambda, T, y, z) based on equation (2.4).

        Parameters:
        lambda_ (float): Wavelength [nm]
        T (float): Temperature [K]
        y (float): y-coordinate [m]
        z (float): z-coordinate [m]
        i (int): Pixel index
        w_y (float): Width of the pixel's photosensitive area along the y-axis [m]
        w_z (float): Width of the pixel's photosensitive area along the z-axis [m]
        delta_y (float): Pixel pitch along the y-axis [m]
        delta_z (float): Pixel pitch along the z-axis [m]

        Returns:
        float: FPA model value [e^- / photons]
        """
        eta_i = self.qe_lambda  # Quantum efficiency at the given wavelength and temperature
          
 

        return eta_i * (self.lambda_ * 1e-9) / (self.h_planck * self.c_light) * self.sqcap(self.w_y, y - i * self.pixel_pitch) * self.sqcap(self.w_z, z - i * self.pixel_pitch)



    def calculate_dark_current(I_d, T):
        """
        Calculate the dark current I_i^d(T) based on equation (2.6).

        Parameters:
        I_d (function): Dark current function I_d(T) [e^- / s]
        T (float): Detector temperature [K]

        Returns:
        float: Calculated dark current I_i^d(T) [e^- / s]
        """
        return I_d(T)

    def calculate_total_current(I_ph, I_d):
        """
        Calculate the total current I_i(T) based on equation (2.6).

        Parameters:
        I_ph (float): Photo current I_i^ph(T) [e^- / s]
        I_d (float): Dark current I_i^d(T) [e^- / s]

        Returns:
        float: Calculated total current I_i(T) [e^- / s]
        """
        return I_ph + I_d

    def calculate_electrons_generated(I, t_int):
        """
        Calculate the number of electrons generated N_i(T) based on equation (2.7).

        Parameters:
        I (float): Total current I_i(T) [e^- / s]
        t_int (float): Integration time [s]

        Returns:
        float: Number of electrons generated N_i(T) [e^-]
        """
        return I * t_int

    def calculate_signal(N, N_o, g):
        """
        Calculate the digital signal S_i(T) based on equation (2.8).

        Parameters:
        N (float): Number of electrons generated N_i(T) [e^-]
        N_o (float): Offset equivalent to the amount of electrons [e^-]
        g (float): Gain [unitless]

        Returns:
        float: Digital signal S_i(T) [DN]
        """
        return g * (N + N_o)

    # Example usage:
    # Define the optical system function O, spectral radiance field L_lambda, and FPA model D_i
    def L_lambda_example(alpha, beta, lambda_):
        return np.sin(alpha) * np.cos(beta) * np.exp(-lambda_)

    def I_d_example(T):
        return 1e-6 * np.exp(-T / 300)
    


    # # Define the ranges for alpha, beta, and lambda
    # alpha_range = (0, np.pi)
    # beta_range = (0, np.pi)
    # lambda_range = (0, 1000)

    # # Define other parameters
    # A_e = 1.0  # m^2
    # A_d = 1.0  # m^2
    # T = 300  # K
    # t_int = 1.0  # s
    # N_o = 100  # e^-
    # g = 1.0  # unitless
    # w_y = 1e-6  # m
    # w_z = 1e-6  # m
    # delta_y = 1e-6  # m
    # delta_z = 1e-6  # m

    # # Calculate the photo current
    # I_ph = calculate_photo_current(O_example, L_lambda_example, D_i, A_e, A_d, alpha_range, beta_range, lambda_range, T, w_y, w_z, delta_y, delta_z)
    # print(f"Photo current: {I_ph} e^- / s")

    # # Calculate the dark current
    # I_d = calculate_dark_current(I_d_example, T)
    # print(f"Dark current: {I_d} e^- / s")

    # # Calculate the total current
    # I_total = calculate_total_current(I_ph, I_d)
    # print(f"Total current: {I_total} e^- / s")

    # # Calculate the number of electrons generated
    # N_generated = calculate_electrons_generated(I_total, t_int)
    # print(f"Electrons generated: {N_generated} e^-")

    # # Calculate the digital signal
    # S = calculate_signal(N_generated, N_o, g)
    # print(f"Digital signal: {S} DN")