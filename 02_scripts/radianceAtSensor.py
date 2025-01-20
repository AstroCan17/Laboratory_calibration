from Py6S import *
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def calculate_total_irradiance(altitude,isrf_fwhm, lambda_min, lambda_max,solar_z_inp,solar_a_inp,view_zenith, view_azimuth):
    
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = solar_z_inp  # Solar zenith angle
    s.geometry.solar_a = solar_a_inp  # Solar azimuth angle
    LOG.info('Py6s initialized'+ "\n" + ("-" * 50) + "\n"+

             f"Default Solar zenith angle: {s.geometry.solar_z} degrees\n"
             f"Default Solar azimuth angle: {s.geometry.solar_a} degrees"
             + "\n" + ("-" * 50))



    # Set the view angles in SixS
    s.geometry.view_z = view_zenith  # View zenith angle
    s.geometry.view_a = view_azimuth  # View azimuth angle

    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(altitude/1000)  # Convert to km
    lambda_min = lambda_min * 1e-3  # Convert to um
    lambda_max = lambda_max * 1e-3  # Convert to um
    isrf_fwhm = isrf_fwhm * 1e-3  # Convert to um

    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    wavelengths, values = SixSHelpers.Wavelengths.run_wavelengths(s, np.arange(lambda_min, lambda_max, isrf_fwhm), output_name='pixel_radiance')

    SixSHelpers.Wavelengths.plot_wavelengths(wavelengths, values, r" At - sensor Spectral Radiance ( $W / m ^2\!/\mu m$ ) " ) 


    LOG.info(("-" * 50) + "\n"+                
            #  f'Total irradiance: {L_lambda:.2f} W/m²/sr/nm\n'
             f'View zenith angle: {view_zenith:.2f} degrees\n'
             f'View azimuth angle: {view_azimuth:.2f} degrees'
             + "\n" + ("-" * 50))
    return wavelengths, values


