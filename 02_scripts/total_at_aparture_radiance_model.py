

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


class TotalAtSensorRadiance:
    def __init__(self):
        self.rayleigh_coefficient = 1.0e-4
        self.mie_coefficient = 1.0e-2

        self.wavelength = 1666.2e-9  # Use lambda = 1666.2 nm for methane
        self.temperature = 5900  # Temperature of the sun in Kelvin
        
        self.spectral_range_min = 1580e-9  # Min wavelength in meters
        self.spectral_range_max = 1690e-9  # Max wavelength in meters
        self.solar_radius = 6.9634e8  # Radius of the Sun in meters
        self.distance_to_earth = 1.496e11  # Average distance from Sun to Earth in meters
        self.surface_albedo = 0.167  # Surface albedo at 1666.2 nm (unitless)

        # Parameters for radiation calculations
        self.atmospheric_transmission = 0.023  # Mean transmission around 1666 nm
        self.incident_angle = 90  # Incident angle in degrees
        self.F_xy = 1  # Assume Lambertian surface

    # Function to calculate spectral radiant exitance using Planck's equation
    def plancks_radiation_law(self, input_wavelength):
        """
        Calculate spectral radiant exitance of the sun using Planck's law (Equation 2-1).
        Input:
        - input_wavelength: Wavelength in meters
        - self.temperature: Temperature of the blackbody in Kelvin (K)
        Output:
        - Spectral radiant exitance M_lambda (W/m^2/um)
        """
        C1 = 3.74151e8  # W*m^2*um^4 (first radiation constant)
        C2 = 1.43879e4  # um*K (second radiation constant)
        return (C1 / (input_wavelength**5)) / (np.exp(C2 / (input_wavelength * self.temperature)) - 1)

    # Function to calculate the top-of-atmosphere irradiance (Equation 2-3)
    def calculate_toa_irradiance(self, specify_wavelength=bool):
        """
        Calculate the top-of-atmosphere irradiance using solar spectral radiant exitance (Equation 2-3).
        Input:
        - self.spectral_range_min: Minimum wavelength in meters
        - self.spectral_range_max: Maximum wavelength in meters
        - self.solar_radius: Radius of the Sun in meters
        - self.distance_to_earth: Average distance from the Sun to Earth in meters
        - specify_wavelength: Boolean indicating whether to specify wavelength range
        Output:
        - Top-of-atmosphere irradiance (W/m^2/um) E_lambda_0
        """
        if specify_wavelength:
            # Average solar radiance over the specified spectral range
            solar_radiance_M_min = self.plancks_radiation_law(self.spectral_range_min)  # W/m^2/m
            solar_radiance_M_max = self.plancks_radiation_law(self.spectral_range_max)  # W/m^2/m
            solar_radiance_M = (solar_radiance_M_min + solar_radiance_M_max) / 2  # Average of min and max
        else:
            solar_radiance_M = self.plancks_radiation_law(self.wavelength)  # W/m^2/m
        area_solar_disk = np.pi * (self.solar_radius ** 2)
        return (solar_radiance_M / np.pi) * (area_solar_disk / (self.distance_to_earth ** 2))

    # Function to calculate the irradiance at the Earth's surface (Equation 2-5)
    def calculate_surface_irradiance(self):
        """
        Calculate the irradiance at the Earth's surface (Equation 2-5).
        This includes atmospheric transmission (solar path transmittance) and cosine factor of incident angle.
        Input:
        - self.atmospheric_transmission: Solar path transmittance (unitless)
        - self.incident_angle: Incident angle of the solar radiation in degrees
        Output:
        - Irradiance at the Earth's surface (W/m^2/um) E_lambda
        """
        toa_irradiance_E_lambda_0 = self.calculate_toa_irradiance(specify_wavelength=True)
        cos_theta = np.cos(np.radians(self.incident_angle))
        surface_irradiance_E_lambda = self.atmospheric_transmission * toa_irradiance_E_lambda_0 * cos_theta
        if surface_irradiance_E_lambda > toa_irradiance_E_lambda_0:
            raise ValueError("Surface irradiance must be less than or equal to top-of-atmosphere irradiance by definition of transmittance.")
        return surface_irradiance_E_lambda

    # Function to calculate the surface radiance (Equation 2-7)
    def calculate_surface_radiance(self):
        """
        Calculate the surface radiance based on Lambertian reflection (Equation 2-7).
        Reflectance is converted using the geometric factor and diffuse spectral reflectance.
        Input:
        - self.surface_albedo: Surface albedo (unitless, between 0 and 1)
        - self.calculate_surface_irradiance(): Irradiance at the Earth's surface E_lambda (W/m^2/um) 
        Output:
        - Surface radiance (W/m^2/um) L_lambda
        """
        surface_irradiance_E_lambda = self.calculate_surface_irradiance()
        return (self.surface_albedo * surface_irradiance_E_lambda) / np.pi

    # Function to calculate the sensor radiance after accounting for atmospheric effects (Equation 2-8)
    def calculate_at_sensor_radiance(self):
        """
        Calculate the radiance reaching the sensor after propagating through the atmosphere (Equation 2-8).
        Input:
        - self.atmospheric_transmission: Atmospheric transmission along the view path (unitless)
        - self.calculate_surface_radiance(): Surface radiance L_lambda (W/m^2/um)
        Output:
        - Radiance at the sensor (W/m^2/um) L_lambda^su
        """
        surface_radiance_L_lambda = self.calculate_surface_radiance()
        return surface_radiance_L_lambda * self.atmospheric_transmission

    # Function to calculate surface-reflected, atmosphere-scattered component (Equation 2-9)
    def calculate_surface_reflected_atmosphere_scattered_component(self):
        """
        Calculate the radiance due to downward scattered radiation from the atmosphere (skylight) (Equation 2-9).
        This term is proportional to the diffuse reflectance and irradiance at the surface due to skylight.
        Input:
        - self.F_xy: Fraction of sky hemisphere visible from the pixel of interest (unitless)
        - self.surface_albedo: Surface albedo (unitless, between 0 and 1)
        - self.atmospheric_transmission: Atmospheric transmission (unitless)
        - self.calculate_surface_irradiance(): Irradiance at the surface due to skylight E_lambda_d (W/m^2/um)
        Output:
        - Surface-reflected, atmosphere-scattered radiance (W/m^2/um/sr) L_lambda^sd
        """
        # Placeholder value for downward irradiance at the surface due to skylight
        E_lambda_d = self.calculate_surface_irradiance() * 0.1  # Assuming skylight contribution is a fraction of direct irradiance
        return self.F_xy * self.surface_albedo * (self.atmospheric_transmission * E_lambda_d) / np.pi

    # Function to calculate path radiance (Equation 2-10)
    def calculate_path_radiance(self, wavelength_m):
        """
        Calculate the radiance due to path-scattered atmospheric components (Equation 2-10).
        Input:
        - wavelength_m: Wavelength in meters
        - self.rayleigh_coefficient: Coefficient for Rayleigh scattering
        - self.mie_coefficient: Coefficient for Mie scattering
        Output:
        - Path radiance (W/m^2/um/sr) L_lambda^sp
        """
        # Convert wavelength to micrometers
        wavelength_um = wavelength_m * 1e6

        # Rayleigh scattering component (proportional to lambda^-4)
        rayleigh_scattering = self.rayleigh_coefficient * wavelength_um**-4

        # Mie scattering component (proportional to lambda^-0.7)
        mie_scattering = self.mie_coefficient * wavelength_um**-0.7

        # Total path radiance as the sum of Rayleigh and Mie scattering
        return rayleigh_scattering + mie_scattering

    # Function to calculate total radiance at the sensor
    def calculate_total_radiance(self, wavelength):
        """
        Calculate the total spectral radiance reaching the sensor (Equation 2-10).
        Includes all major components: surface-reflected, scattered, and path radiance.
        Input:
        - surface_radiance: Surface-reflected radiance (W/m^2/um/sr)
        - E_lambda_d: Downward scattered irradiance (W/m^2/um)
        - wavelength: Wavelength in meters
        - F_xy: Fraction of sky hemisphere visible from the pixel of interest (unitless)
        - atmospheric_transmission: Atmospheric transmission (unitless)
        Output:
        - Total radiance at the sensor (W/m^2/um/sr) L_lambda^s
        """
        L_surface = self.calculate_at_sensor_radiance()
        L_atmosphere = self.calculate_surface_reflected_atmosphere_scattered_component()
        L_path = self.calculate_path_radiance(wavelength)
        return L_surface + L_atmosphere + L_path

    def run_atmospheric_simulator(self):
        # Initialize the TotalAtSensorRadiance class
        total_radiance = TotalAtSensorRadiance()
        
        wavelength = 1666.2e-9  # Wavelength for methane absorption line
        total_radiance_value = total_radiance.calculate_total_radiance(wavelength)
        print("-" * 50)
        print(f"Total radiance at sensor for wavelength {wavelength * 1e9:.2f} nm:{total_radiance_value:.2e} W/m^2/um/sr")
        print("-" * 50)
        return total_radiance_value
        

if __name__ == "__main__":
    total_radiance = TotalAtSensorRadiance()
    total_radiance.run_atmospheric_simulator()