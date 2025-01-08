"""
Photon Count Simulator for C-Red NS Sensor
------------------------------------------
Developed by: Can Deniz Kaya
Date: 2025-01-04

"""



import numpy as np

class PhotonCountSimulator: 
    """
    This class models a C-Red NS sensor where only the along-track direction
    (forward/backward motion) is off-nadir, while across-track (sideways)
    is kept at a nadir-like geometry. 
    """
    
    def __init__(
        self,
        altitude_m=float,       # Satellite altitude [m] (e.g. 705 km)
        off_nadir_deg=float,      # Off-nadir angle in the along-track direction [degrees]
        L_toa=float,              # TOA radiance [W / m^2 / sr / nm]
        
        lambda_min_nm=int,       # Minimum wavelength [nm]
        lambda_max_nm=int,       # Maximum wavelength [nm]

        t_int=float               # Integration time [s]

    ):
        """
        Constructor for the PhotonCountSimulator class.

        Parameters
        ----------
        altitude_m : float
            Satellite altitude in meters.
        off_nadir_deg : float
            Off-nadir angle in degrees (applied only along-track).
        L_toa : float
            Top-of-atmosphere radiance in W / m^2 / sr / nm.
        OE : float
            Optical efficiency (fraction, dimensionless).
        lambda_min_nm : float
            Minimum wavelength in nanometers.
        lambda_max_nm : float
            Maximum wavelength in nanometers.
        quantum_eff : float
            Quantum efficiency (dimensionless, e.g. 0.9 => 90%).
        t_int : float
            Integration time in seconds.
        dark_current_rate : float
            Dark current rate per pixel per second [e- / pixel / s].
        read_noise_e : float
            Read noise in electrons RMS per pixel [e- rms / pixel].
        focal_length_m : float
            Camera focal length in meters.
        pixel_size_m : float
            Physical pixel size in meters (e.g. 15e-6 = 15 micrometers).
        n_pixels_across : int
            Number of pixels in the across-track (x) direction.
        n_pixels_along : int
            Number of pixels in the along-track (y) direction.
        """

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
        self.lambda_min  = float(lambda_min_nm)
        self.lambda_max  = float(lambda_max_nm)
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

    ###########################################################################
    # 1) ORBITAL AND GEOMETRIC FUNCTIONS
    ###########################################################################
    def calc_slant_range(self) -> float:
        """
        Computes the slant range [m] considering off-nadir angle
        using the law of cosines for a spherical Earth model:
        
        slant_range = sqrt( (R + h)^2 + R^2 - 2 * R * (R + h) * cos(alpha) )

        Returns
        -------
        float
            Slant range in meters [m].
        """
        alpha_rad = np.radians(self.off_nadir_deg)
        R = self.R_earth
        h = self.altitude_m
        return np.sqrt((R + h)**2 + R**2 - 2.0*R*(R + h)*np.cos(alpha_rad))
    
    def calc_orbital_velocity(self) -> float:
        """
        Computes the orbital velocity [m/s] assuming a circular orbit:

        v_orbit = sqrt( mu / (R + h) )

        Returns
        -------
        float
            Orbital velocity in meters per second [m/s].
        """
        return np.sqrt(self.mu / (self.R_earth + self.altitude_m))
    
    def calc_groundtrack_velocity(self) -> float:
        """
        Computes ground track velocity [m/s] of the satellite sub-point.

        ground_track_velocity = R_earth * (orbital_velocity / (R_earth + altitude))

        Returns
        -------
        float
            Ground track velocity in meters per second [m/s].
        """
        v_orbit = self.calc_orbital_velocity()
        r_plus_h = self.R_earth + self.altitude_m
        angular_rate = v_orbit / r_plus_h
        return self.R_earth * angular_rate

    ###########################################################################
    # 2) GSD (ACROSS-TRACK CONSTANT, ALONG-TRACK OFF-NADIR)
    ###########################################################################
    def calc_gsd_across(self) -> float:
        """
        Computes the across-track GSD [m], assumed constant (nadir geometry).

        GSD_across = (altitude * pixel_size) / focal_length

        Returns
        -------
        float
            Across-track GSD in meters [m].
        """
        return (self.altitude_m * self.pixel_size) / self.f_m

    def calc_gsd_along(self) -> float:
        """
        Computes the along-track GSD [m], which changes with off-nadir angle.
        The slant range is computed via the law of cosines.

        Basic approach:
            d_slant = calc_slant_range()
            GSD_along = ( d_slant * pixel_size ) / focal_length

        Returns
        -------
        float
            Along-track GSD in meters [m].
        """
        d_slant = self.calc_slant_range()
        gsd_simple = (d_slant * self.pixel_size) / self.f_m
        alpha_rad = np.radians(self.off_nadir_deg)
        return gsd_simple / np.cos(alpha_rad)
    

    def calc_time_for_alongtrack_gsd(self) -> float:
        """
        Computes the time to travel the along-track GSD [s] at the given off-nadir angle.

        time_for_gsd = GSD_along / ground_track_velocity

        Returns
        -------
        float
            Time in seconds to travel the along-track GSD.
        """
        gsd_y = self.calc_gsd_along()
        gt_v = self.calc_groundtrack_velocity()
        return gsd_y / gt_v

    def calc_swath_across(self) -> float:
        """
        Computes the across-track swath [m].

        swath_across = calc_gsd_across() * nx

        Returns
        -------
        float
            Across-track swath width in meters [m].
        """
        return self.calc_gsd_across() * self.nx
    
    def calc_swath_along(self) -> float:
        """
        Computes the along-track swath [m].

        swath_along = calc_gsd_along() * ny

        Returns
        -------
        float
            Along-track swath length in meters [m].
        """
        return self.calc_gsd_along() * self.ny

    ###########################################################################
    # 3) AREA, LINE RATE, FRAME RATE
    ###########################################################################
    def calc_line_rate_s(self) -> float:
        """
        Computes the line rate [s/line] in the along-track direction.

        line_rate = GSD_along / ground_track_velocity

        Returns
        -------
        float
            Time in seconds per line [s/line].
        """
        gsd_y = self.calc_gsd_along()
        gt_v = self.calc_groundtrack_velocity()
        return gsd_y / gt_v
    
    def calc_frame_rate_hz(self) -> float:
        """
        Computes the frame rate [Hz].

        frame_rate = 1 / (line_rate_s * ny)

        Returns
        -------
        float
            Frame rate in Hertz [frames/s].
        """
        line_rate = self.calc_line_rate_s()
        return 1.0 / (line_rate * self.ny)
    
    def calc_area_during_time(self, time_s: float) -> float:
        """
        Computes the surface area [m^2] covered by the sensor in a given time
        (assuming across-track is constant, along-track coverage depends on
        satellite's ground track motion).

        area = swath_across [m] * (ground_track_velocity [m/s] * time_s [s])

        Parameters
        ----------
        time_s : float
            Time duration in seconds over which area is covered.

        Returns
        -------
        float
            Covered area in square meters [m^2].
        """
        swath_x = self.calc_swath_across()     # [m]
        gt_v = self.calc_groundtrack_velocity()# [m/s]
        along_track_dist = gt_v * time_s       # [m]
        return swath_x * along_track_dist      # [m^2]

    ###########################################################################
    # 4) RADIOMETRIC CALCULATIONS
    ###########################################################################
    def calc_solid_angle(self, slant_range_m: float) -> float:
        """
        Computes the solid angle [sr] of the sensor aperture given the slant range.

        Omega = pi * (aperture_radius^2) / (slant_range^2)

        Parameters
        ----------
        slant_range_m : float
            Slant range in meters [m].

        Returns
        -------
        float
            Solid angle in steradians [sr].
        """
        aperture_diam = float(0.072) # m   KTO design 
        r_ap = aperture_diam / 2.0
        return np.pi * (r_ap**2) / (slant_range_m**2)
    
    def calc_power_detected(self, slant_range_m: float, surface_area_m2: float) -> float:
        """
        Computes the optical power [W] detected by the sensor.

        P_detector = L_toa [W/m^2/sr/nm]
                     * Omega [sr]
                     * surface_area_m2 [m^2]
                     * OE [dimensionless]
                     * delta_lambda_nm [nm]

        Parameters
        ----------
        slant_range_m : float
            Slant range [m].
        surface_area_m2 : float
            Surface area [m^2] contributing to the radiance.

        Returns
        -------
        float
            Power in Watts [W].
        """
        Omega = self.calc_solid_angle(slant_range_m)
        return (self.L_toa 
                * Omega 
                * surface_area_m2
                * self.OE
                * self.delta_lambda_nm)
    
    def calc_photoelectrons(self, power_W: float) -> float:
        """
        Converts the detected power [W] into number of photoelectrons, given 
        the integration time and quantum efficiency.

        N_e = power_W [W]
              * t_int [s]
              * QE [dimensionless]
              * (lambda [m] / (h_planck [J*s] * c_light [m/s]))

        Using the center wavelength lambda_c_nm for conversion.

        Parameters
        ----------
        power_W : float
            Power in Watts [W].

        Returns
        -------
        float
            Total photoelectrons [dimensionless].
        """
        lam_m = (self.lambda_c_nm * 1e-9)  # Convert nm -> m
        factor = lam_m / (self.h_planck * self.c_light)
        return power_W * self.t_int * self.QE * factor
    
    def calc_snr_total(self, N_signal_total: float) -> float:
        """
        Computes the SNR (signal-to-noise ratio) considering shot noise, 
        dark current, and read noise for the entire detector.

        SNR = N_signal_total 
              / sqrt( N_signal_total + N_dark_total + (RN_total^2) )

        Where:
          - N_dark_total = dark_current_rate [e-/pix/s] 
                           * t_int [s] 
                           * (nx * ny)  (all pixels)
          - RN_total ~ read_noise_e [e- rms/pix] * sqrt(nx * ny)

        Parameters
        ----------
        N_signal_total : float
            Total photoelectrons from the entire detector [dimensionless].

        Returns
        -------
        float
            SNR [dimensionless].
        """
        total_pix = self.nx * self.ny
        
        N_dark_total = self.dark_current_rate * self.t_int * total_pix
        RN_total = self.read_noise_e * np.sqrt(total_pix)
        
        noise_var = N_signal_total + N_dark_total + (RN_total**2)
        if noise_var <= 0:
            return 0.0
        return N_signal_total / np.sqrt(noise_var)
    
    ###########################################################################
    # 5) MAIN METHOD: RUN SIMULATION
    ###########################################################################
    def run_alongtrack_simulation(self, time_s):
        """
        Performs the complete chain of calculations for an along-track 
        off-nadir scenario. 

        1) Compute slant range.
        2) Compute across- and along-track GSDs.
        3) Compute swaths, line rate, frame rate.
        4) Compute the area covered in `time_s`.
        5) Compute detected power and photoelectrons.
        6) Compute overall SNR (assuming total-detector approach).

        Parameters
        ----------
        time_s : float
            Duration (in seconds) during which the area is covered.

        Returns
        -------
        dict
            Dictionary of results including geometry, radiometric, and SNR information.
        """
        if time_s is None:
            # Compute time to travel along-track GSD
            time_s = self.calc_time_for_alongtrack_gsd()
        else:
            time_s = float(time_s)
        # Slant range
        d_slant = self.calc_slant_range()
        
        # GSD
        gsd_x = self.calc_gsd_across()
        gsd_y = self.calc_gsd_along()
        
        # Swaths
        swath_x = self.calc_swath_across()
        swath_y = self.calc_swath_along()
        
        # line_rate, frame_rate
        line_rate_s = self.calc_line_rate_s()
        frame_rate_hz = self.calc_frame_rate_hz()
        
        # Area covered in `time_s`
        area_m2 = self.calc_area_during_time(time_s)
        
        # Power -> photoelectrons
        power_in_W = self.calc_power_detected(d_slant, area_m2)
        N_electrons = self.calc_photoelectrons(power_in_W)
        
        # SNR
        snr_val = self.calc_snr_total(N_electrons)
        
        return {
            "altitude_km": self.altitude_m / 1e3,
            "off_nadir_deg": self.off_nadir_deg,
            "slant_range_km": d_slant / 1e3,
            "gsd_across_m": gsd_x,
            "gsd_along_m": gsd_y,
            "simulation_time_s": time_s,
            "integration_time_s": self.t_int,
            "swath_across_km": swath_x/1000,
            "swath_along_km": swath_y/1000,
            "line_rate_s": line_rate_s,
            "frame_rate_hz": frame_rate_hz,
            "area_km2_in_time_s": area_m2*1e-6,
            "power_in_W": power_in_W,
            "N_electrons": N_electrons,
            "SNR": snr_val
        }


if __name__ == "__main__":
    # Example usage testing different off-nadir angles
    answer = input("Do you want to set simulation time manually? (y/n): ")
    if answer == "y":
        simulation_time = float(input("Enter simulation time in seconds: "))
        simulation_time = float(simulation_time)
    else:
        print("Simulation time will be calculated automatically by using ground track velocity and along track GSD req.")
        simulation_time = None
    for angle in [0, 10, 20, 30]:
        sat = PhotonCountSimulator(
            altitude_m=500e3,
            off_nadir_deg=angle,
            L_toa=0.01,
            lambda_min_nm=1580,
            lambda_max_nm=1690,
            t_int=0.1
        )
        results = sat.run_alongtrack_simulation(time_s=simulation_time)
        
        print(f"\n--- OFF-NADIR: {angle} deg ---")
        for k, v in results.items():
            print(f"{k}: {v:.4g}")
    
    print("\nSimulation completed.")
    print("Goodbye blue sky!")