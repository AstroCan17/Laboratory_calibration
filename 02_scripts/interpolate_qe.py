
from scipy.interpolate import interp1d
import pandas as pd
import os 



class QeInterpolation:
    def __init__(self, T, spectral_responsivity_path):
        self.T = T
        self.spectral_responsivity_path = spectral_responsivity_path
        pass
            
    def get_spectral_responsivity(self,spectral_responsivity_path):
        """
        Import spectral responsivity data from the given CSV file.

        Parameters:
        - spectral_responsivity_path: Path to the CSV file containing spectral responsivity data

        Returns:
        - wavelengths: List of wavelengths in meters
        - efficiencies: List of efficiencies corresponding to each wavelength
        """
        # Load the spectral responsivity data from CSV
        df = pd.read_csv(spectral_responsivity_path, names=['Data'])
        wavelengths = []
        efficiencies = []
        for data in df['Data']:
            wavelength, efficiency = data.split(',')
            wavelengths.append(float(wavelength) * 1e-9)  # Convert nm to meters
            efficiencies.append(float(efficiency))  

        return wavelengths, efficiencies


    def interpolate_efficiencies(self):
        """
        Interpolate efficiencies for a target temperature based on data from 5 and 35 degrees.

        Parameters:
        - target_temperature: Temperature to interpolate to (default is 20 degrees)

        Returns:
        - wavelengths_common: List of common wavelengths
        - efficiencies_target: List of interpolated efficiencies at the target temperature
        """
        # Load spectral responsivity data
        target_temperature = self.T
        for file_csv in os.listdir(self.spectral_responsivity_path):
            if file_csv.endswith(".csv"):
                if "_5" in file_csv:
                    spectral_responsivity_path_5 = os.path.join(self.spectral_responsivity_path, file_csv)
                elif "_35" in file_csv:
                    spectral_responsivity_path_35 = os.path.join(self.spectral_responsivity_path, file_csv)

        wavelengths_5, efficiencies_5 = self.get_spectral_responsivity(spectral_responsivity_path_5)
        wavelengths_35, efficiencies_35 = self.get_spectral_responsivity(spectral_responsivity_path_35)

        # Find the common wavelength range between the two datasets
        wavelengths_common = sorted(set(wavelengths_5).union(set(wavelengths_35)))

        # Interpolating the efficiencies at 5 and 35 degrees to cover the entire wavelength range
        efficiency_interp_5 = interp1d(wavelengths_5, efficiencies_5, kind='cubic', fill_value="extrapolate")
        efficiency_interp_35 = interp1d(wavelengths_35, efficiencies_35, kind='cubic', fill_value="extrapolate")

        # Calculate efficiencies for the common wavelengths
        efficiencies_5_at_common = efficiency_interp_5(wavelengths_common)
        efficiencies_35_at_common = efficiency_interp_35(wavelengths_common)

        # Interpolating to find spectral responsivity at the target temperature
        efficiencies_target = efficiencies_5_at_common + (target_temperature - 5) / (35 - 5) * (efficiencies_35_at_common - efficiencies_5_at_common)
        # Check if the file already exists
        output_file_name = f'interpolated_data_{int(self.T)}.csv'
        output_file = self.spectral_responsivity_path+"/"+ output_file_name
        if not os.path.exists(output_file):
            # Save the interpolated data
            df = pd.DataFrame({'Wavelength': wavelengths_common, 'Efficiency': efficiencies_target})
            df.to_csv(output_file, index=False)
        return wavelengths_common, efficiencies_target


def run_interpolation(T, spectral_responsivity_path):
    qe_interpolation = QeInterpolation(T, spectral_responsivity_path)

    wavelengths_common, efficiencies_target = qe_interpolation.interpolate_efficiencies()
    return wavelengths_common, efficiencies_target
    