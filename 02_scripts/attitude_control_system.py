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

    def euler_to_matrix_ned(self, degrees=True):
        """
        NED (North-East-Down) eksenlerinde:
        roll  => x(North) ekseni etrafında dönüş
        pitch => y(East)  ekseni etrafında dönüş
        yaw   => z(Down)  ekseni etrafında dönüş
        
        Dönüş sırası (intrinsic rotations): Rz(yaw) * Ry(pitch) * Rx(roll).
        
        Parametreler:
        -----------
        roll, pitch, yaw : float
            Euler açıları
        degrees : bool, default=True
            True ise açıların derece cinsinden verildiği kabul edilir,
            fonksiyon içinde radyana dönüştürülür.
        
        Çıktı:
        ------
        R : numpy.ndarray (3x3)
            Toplam dönüş (rotasyon) matrisi.
        """
        if degrees:
            roll  = np.deg2rad(self.roll_angle)
            pitch = np.deg2rad(self.pitch_angle)
            yaw   = np.deg2rad(self.yaw_angle)
        
        # Rx(roll): x(North) ekseni etrafında
        Rx = np.array([
            [1,           0,            0],
            [0,  np.cos(roll), -np.sin(roll)],
            [0,  np.sin(roll),  np.cos(roll)]
        ])
        
        # Ry(pitch): y(East) ekseni etrafında
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0,              1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rz(yaw): z(Down) ekseni etrafında
        Rz = np.array([
            [ np.cos(yaw), -np.sin(yaw), 0],
            [ np.sin(yaw),  np.cos(yaw), 0],
            [           0,            0, 1]
        ])
        
        # Toplam dönüş matrisi
        R = Rz @ Ry @ Rx
        return R

    def compute_alpha_beta(self, degrees=True):
        """
        1) Tezdeki incidence açıları (alpha, beta) 
        alpha = arctan2(v_z, v_x)
        beta  = arctan2(v_z, v_y)
        
        2) View Zenith (theta_v) ve View Azimuth (phi_v) 
        theta_v = arccos( v_z / |v| )
        phi_v   = arctan2(v_y, v_x)
        
        Uygulanan koordinat sistemi: NED (x=North, y=East, z=Down).
        
        Parametreler:
        -----------
        roll, pitch, yaw : float
            Euler açıları (NED tanımına göre).
        degrees : bool, default=True
            True ise roll, pitch, yaw açıları derece cinsinden olduğu varsayılır
            ve sonuçlar da derece cinsinden döndürülür.
        
        Çıktı:
        ------
        alpha, beta, theta_v, phi_v : float
            - alpha, beta     => Tezdeki 'incidence' açıları
            - theta_v, phi_v  => View zenith ve view azimuth açıları
        """
        # 1) Dönüş matrisi
        R = self.euler_to_matrix_ned(degrees=degrees)
        
        # 2) Kameranın boresight vektörü: Down yönü (z) => (0, 0, 1)
        v_s = np.array([0.0, 0.0, 1.0])
        
        # 3) Dönüşten sonra elde edilen bakış yönü
        v_e = R @ v_s  # [v_x, v_y, v_z] (NED çerçevesinde)
        
        # Büyüklük
        mag_v = np.linalg.norm(v_e)
        v_x, v_y, v_z = v_e
        
        # ------------------------
        # A) Tezdeki alpha, beta
        # alpha = arctan2(v_z, v_x)
        # beta  = arctan2(v_z, v_y)
        alpha_rad = np.arctan2(v_z, v_x)
        beta_rad  = np.arctan2(v_z, v_y)
        
        # ------------------------
        # B) View Zenith & View Azimuth
        # theta_v = arccos(v_z / |v|)
        # phi_v   = arctan2(v_y, v_x)  (North'tan East'e doğru)
        theta_v_rad = np.arccos(v_z / mag_v)
        phi_v_rad   = np.arctan2(v_y, v_x)
        
        if degrees:
            alpha   = np.rad2deg(alpha_rad)
            beta    = np.rad2deg(beta_rad)
            theta_v = np.rad2deg(theta_v_rad)
            phi_v   = np.rad2deg(phi_v_rad)
            azimuth = phi_v
            zenith = theta_v
            LOG.info(("-" * 50) + "\n" +
                     f"Incidence angles:\n"
                     f"Along Track Alpha: {alpha:.2f} degrees\n"
                     f"Across Track Beta: {beta:.2f} degrees\n"
                     f"View Zenith: {zenith:.2f} degrees\n"
                     f"View Azimuth: {azimuth:.2f} degrees\n"
                     + ("-" * 50))
            return alpha, beta, zenith, azimuth
        else:
            LOG.info(("-" * 50) + "\n" +
                     f"Incidence angles:\n"
                     f"Along Track Alpha: {alpha_rad:.2f} radians\n"
                     f"Across Track Beta: {beta_rad:.2f} radians\n"
                     f"View Zenith: {theta_v_rad:.2f} radians\n"
                     f"View Azimuth: {phi_v_rad:.2f} radians\n"
                     + ("-" * 50))
            return alpha_rad, beta_rad, theta_v_rad, phi_v_rad


