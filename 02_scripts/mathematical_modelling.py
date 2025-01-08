
import numpy as np
import




class TheorySec2:
    
    def __init__(
        self,
        altitude_m=float,         # Satellite altitude [m] (e.g. 705 km)
        off_nadir_deg=float,      # Off-nadir angle in the along-track direction [degrees]
        L_toa=float,              # TOA radiance [W / m^2 / sr / nm]
        t_int=float,              # Integration time [s]
        o_optical_fwm = None

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
        self.altitude_m    = float(altitude_m)
        self.off_nadir_deg = off_nadir_deg
        
        # Radiometric params
        self.L_toa       = L_toa                 
        self.OE          = 0.6          # Optical efficiency [dimensionless] KTO design worst case scenario

        self.lambda_min  = float(1580)  # [nm] Minimum wavelength
        self.lambda_max  = float(1690)  # [nm] Maximum wavelength
        self.delta_lambda_nm = (self.lambda_max - self.lambda_min)
        self.QE          = 0.18         # Quantum efficiency [dimensionless] for lambda_c_nm = 1666.2 nm
        self.t_int       = t_int                # Integration time [s]
        
        # Noise params
        self.dark_current_rate = 600      # [e- / pix / s]
        self.read_noise_e      = 30       # [e- rms / pix]
        
        # Optic/detector
        self.f_m         = 0.18               # [m] focal length KTO design
        self.pixel_size  = 15e-6              # [m]
            # Detector dimensions (640 x 512), but 2 pixels are dead referred product's manual 
        self.nx          = 638    # pixel count (across-track)
        self.ny          = 510     # pixel count (along-track)
        
        # "center" wavelength in nm for methane detection
        self.lambda_c_nm = 1666.2             # nm (e.g. for methane detection)


        # Optics Forward model parameters and functions

        self.o_optical() = o_optical_fwm


        pass



    def calculate_irradiance_simplified(self,O, L_lambda, A_e, alpha_range, beta_range, lambda_range):
        """
        Calculate the irradiance E(y, z) based on the simplified equation (2.2).

        Parameters:
        O (function): Optical system function O(y, z, alpha, beta, lambda) [unitless]
        L_lambda (function): Spectral radiance field L_lambda(alpha, beta, lambda) [W m^-2 nm^-1 sr^-1]
        A_e (float): Entrance aperture area [m^2]
        alpha_range (tuple): Range of alpha values (min, max) [radians]
        beta_range (tuple): Range of beta values (min, max) [radians]
        lambda_range (tuple): Range of lambda values (min, max) [nm]

        Returns:
        float: Calculated irradiance E(y, z) [W m^-2]
        """
        alpha_values  = np.linspace(alpha_range[0], alpha_range[1], 100)
        beta_values   = np.linspace(beta_range[0], beta_range[1], 100)
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], 100)

        E_yz = 0
        for alpha in alpha_values:
            for beta in beta_values:
                for lambda_ in lambda_values:
                    E_yz += O(alpha, beta, lambda_) * L_lambda(alpha, beta, lambda_)

        E_yz *= A_e
        return E_yz

    # Example usage:
    # Define the optical system function O and spectral radiance field L_lambda
    def O_example(self,alpha, beta, lambda_):
        return np.exp(-alpha**2 - beta**2 - lambda_**2)

    def L_lambda_example(self,alpha, beta, lambda_):
        return np.sin(alpha) * np.cos(beta) * np.exp(-lambda_)

    # Define the ranges for alpha, beta, and lambda
    alpha_range = (0, np.pi)
    beta_range = (0, np.pi)
    lambda_range = (0, 1000)

    # Calculate the irradiance
    E_yz_simplified = calculate_irradiance_simplified(O_example, L_lambda_example, 1.0, alpha_range, beta_range, lambda_range)
    print(E_yz_simplified)


