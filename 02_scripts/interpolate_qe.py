
from scipy.interpolate import interp1d
import pandas as pd
import os 
import matplotlib.pyplot as plt
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
__name__ = "Quantum Efficiency Interpolation"
LOG = logging.getLogger(__name__)



class QeInterpolation:
    def __init__(self, T, spectral_responsivity_path,lambda_min,lambda_max,outputname):
        self.T = T
        self.spectral_responsivity_path = spectral_responsivity_path
        self.lambda_min = float(lambda_min)
        self.lambda_max  = float(lambda_max)
        self.outputname = outputname
      
            
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
            wavelengths.append(float(wavelength))  # Convert nm to meters
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
        
        #find the minimum and maximum temperature from the end of the file name
        min_temp = int(spectral_responsivity_path_5.split("_")[-1].split(".")[0])
        max_temp = int(spectral_responsivity_path_35.split("_")[-1].split(".")[0])

        # Find the common wavelength range between the two datasets
        wavelengths_common = sorted(set(wavelengths_5).union(set(wavelengths_35)))

        # Interpolating the efficiencies at 5 and 35 degrees to cover the entire wavelength range
        efficiency_interp_5 = interp1d(wavelengths_5, efficiencies_5, kind='cubic', fill_value="extrapolate")
        efficiency_interp_35 = interp1d(wavelengths_35, efficiencies_35, kind='cubic', fill_value="extrapolate")

        # Calculate efficiencies for the common wavelengths
        efficiencies_5_at_common = efficiency_interp_5(wavelengths_common)
        efficiencies_35_at_common = efficiency_interp_35(wavelengths_common)

        # Interpolating to find spectral responsivity at the target temperature
        efficiencies_target = efficiencies_5_at_common + (target_temperature - min_temp) / (max_temp - min_temp) * (efficiencies_35_at_common - efficiencies_5_at_common)
        
        # Check if the file already exists
        interpolated_qe = {'Wavelength': wavelengths_common, 'Efficiency': efficiencies_target}
        output_file_name = f'interpolated_data_{int(self.T)}.csv'
        output_file = self.spectral_responsivity_path+"/"+ output_file_name
        if not os.path.exists(output_file):
            # Save the interpolated data
            df = pd.DataFrame({'Wavelength': wavelengths_common, 'Efficiency': efficiencies_target})
            df.to_csv(output_file, index=False)
        
        return interpolated_qe


    def filter_qe(self):
        interpolated_qe = self.interpolate_efficiencies()
        filtered_qe = {
            'Wavelength': [w for w in interpolated_qe['Wavelength'] if self.lambda_min <= w <= self.lambda_max],
            'Efficiency': [e for w, e in zip(interpolated_qe['Wavelength'], interpolated_qe['Efficiency']) if self.lambda_min <= w <= self.lambda_max]
        }
        # save the filtered data
        output_file_name = self.outputname
        output_file = self.spectral_responsivity_path+"/"+ output_file_name
        if not os.path.exists(output_file):
            df = pd.DataFrame({'Wavelength': filtered_qe['Wavelength'], 'Efficiency': filtered_qe['Efficiency']})
            df.to_csv(output_file, index=False)
        return filtered_qe



def run_interpolation(T, spectral_responsivity_path,lambda_min,lambda_max,outputname,visualize=bool):
    LOG.info(('-'*50)+'\n'+
        f"Interpolating quantum efficiency data for temperature {T} °C")
    qe_interpolation = QeInterpolation(T, spectral_responsivity_path,lambda_min,lambda_max,outputname)
    
    interpolated_qe = qe_interpolation.interpolate_efficiencies()
    LOG.info(f"Interpolated QE data saved to {spectral_responsivity_path}/interpolated_data_{int(T)}.csv")
    filtered_qe = qe_interpolation.filter_qe()

    LOG.info((('-'*50)+'\n')+
             f"Quantum Efficiency data filtered for wavelength range {int(lambda_min)} - {int(lambda_max)} nm\n"+
             f"Filtered QE data saved to {spectral_responsivity_path}/filtered_data_{int(T)}.csv\n"+('-'*50))   

    if visualize:
        # Visualize the interpolated QE
        LOG.info('Visualizing the interpolated and filtered QE data')
        plt.figure()
        plt.plot(interpolated_qe['Wavelength'], interpolated_qe['Efficiency'], label='Interpolated QE')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Efficiency [%]')
        plt.title('Interpolated QE')
        plt.legend()
        plt.grid()

        LOG.info('Visualizing the filtered QE data'+ "\n" + ("-" * 50))
        plt.figure()
        plt.plot(filtered_qe['Wavelength'], filtered_qe['Efficiency'], label='Filtered QE')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Efficiency [%]')
        plt.title('Filtered QE')
        plt.legend()
        plt.grid()
        plt.show()
    return filtered_qe

