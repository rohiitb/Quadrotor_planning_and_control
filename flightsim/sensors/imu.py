import numpy as np
from scipy.spatial.transform import Rotation
import math


class Imu:
    """
    Simuated IMU measurement given
      1) quadrotor's ground truth state and acceleration, and
      2) IMU's pose in quadrotor body frame.
    """
    def __init__(self, R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, gravity_vector):
        """
        Parameters:
            R_BS, the rotation from sensor frame S to body frame B
            p_BS, the position vector from frame B's origin to frame S's origin, expressed in frame B
            accelerometer_noise_density, accelerometer "white noise",  [ m / s^2 / sqrt(Hz) ]   
            accelerometer_random_walk, accelerometer bias diffusion,  [ m / s / sqrt(Hz) ]
            gyroscope_noise_density, gyro "white noise",  [ rad / s / sqrt(Hz) ]   
            gyroscope_random_walk, gyro bias diffusion,  [ rad / sqrt(Hz) ]
            sampling_rate, the sampling rate of the sensor, Hz (1/s)
            gravity_vector, the gravitational vector in world frame (should be ~ [0, 0 , -9.81])
        """

        # A few checks
        if type(R_BS) != np.ndarray:
            raise TypeError("R_BS's type is not numpy.ndarray")
        if type(p_BS) != np.ndarray:
            raise TypeError("p_BS's type is not numpy.ndarray")
        if type(gravity_vector) != np.ndarray:
            raise TypeError("gravity_vector's type is not numpy.ndarray")
        if R_BS.shape != (3, 3):
            raise ValueError("R_BS's size is not (3, 3)")
        if p_BS.shape != (3,):
            raise ValueError("p_BS's size is not (3,)")
        if gravity_vector.shape != (3,):
            raise ValueError("gravity_vector's size is not (3,)")

        self.R_BS = R_BS
        self.p_BS = p_BS
        self.gravity_vector = gravity_vector

        self.accelerometer_noise_density = accelerometer_noise_density
        self.accelerometer_random_walk = accelerometer_random_walk
        self.gyroscope_noise_density = gyroscope_noise_density
        self.gyroscope_random_walk = gyroscope_random_walk
    
    def get_noise_parameters(self):
        """
        Outputs:
            accelerometer_noise_density, gyro "white noise",  [ m / s^2 / sqrt(Hz) ]   
            accelerometer_random_walk, gyro bias diffusion,  [ m / s / sqrt(Hz) ]
            gyroscope_noise_density, accel "white noise",  [ rad / s / sqrt(Hz) ]   
            gyroscope_random_walk, accel bias diffusion,  [ rad / sqrt(Hz) ]
        """
        return self.accelerometer_noise_density, self.accelerometer_random_walk, self.gyroscope_noise_density, self.gyroscope_random_walk
        # return self.accelerometer_noise_density * 100, self.accelerometer_random_walk * 100, self.gyroscope_noise_density * 100, self.gyroscope_random_walk * 100

    def measurement(self, state, acceleration, with_noise=True):
        """
        Computes and returns the IMU measurement at a time step.

        Inputs:
            state, a dict describing the state with keys
                x, position, m, shape=(3,)
                v, linear velocity, m/s, shape=(3,)
                q, quaternion [i,j,k,w], shape=(4,)
                w, angular velocity (in LOCAL frame!), rad/s, shape=(3,)
            acceleration, a dict describing the acceleration with keys
                vdot, quadrotor's linear acceleration expressed in world frame, m/s^2, shape=(3,)
                wdot, quadrotor's angular acceleration expressed in LOCAL frame, rad/s^2, shape=(3,)
        Outputs:
            accelerometer_measurement, simulated accelerometer measurement, m/s^2, shape=(3,)
            gyroscope_measurement, simulated gyroscope measurement, rad/s^2, shape=(3,)
        """
        q_WB = state['q']
        w_WB_B = state['w']
        alpha_WB_B = acceleration['wdot']
        a_WB_W = acceleration['vdot']

        # Rotation matrix of the body frame B in world frame W
        R_WB = Rotation.from_quat(q_WB).as_matrix()

        # Sensor position in body frame expressed in world coordinates
        p_BS_W = R_WB @ self.p_BS

        # Linear acceleration of point S (the imu) expressed in world coordinates W.
        w_WB_W = R_WB @ w_WB_B
        alpha_WB_W = R_WB @ alpha_WB_B 
        a_WS_W = a_WB_W + np.cross(alpha_WB_W, p_BS_W) + np.cross(w_WB_W, np.cross(w_WB_W, p_BS_W))

        # Rotation from world to imu: R_SW = R_SB * R_BW
        R_SW = self.R_BS.T @ R_WB.T

        # Rotate to local frame
        accelerometer_measurement = R_SW @ (a_WS_W - self.gravity_vector)
        gyroscope_measurement = self.R_BS.T @ w_WB_B

        # Add noises
        if with_noise:
            accelerometer_measurement += np.random.normal(scale=self.accelerometer_noise_density)
            gyroscope_measurement += np.random.normal(scale=self.gyroscope_noise_density)
        
        return accelerometer_measurement, gyroscope_measurement
