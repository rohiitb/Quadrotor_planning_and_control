import unittest
import numpy as np
import math

from flightsim.sensors.imu import Imu

class SensorAtOriginTestCase(unittest.TestCase):
    def setUp(self):
        gravity_vector = np.array([0, 0 , -9.81])
        
        R_BS = np.identity(3)
        p_BS = np.zeros(3)
        accelerometer_noise_density = 0.0
        accelerometer_random_walk = 0.0
        gyroscope_noise_density = 0.0
        gyroscope_random_walk = 0.0
        self.imu = Imu(R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, gravity_vector)

    def test_stationary(self):
        # Completley stationary.
        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array([0, 0, 0, 1]),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([0,0,9.81])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')

    def test_stationary_tilt(self):
        # Stationary but pitch forward by 90 degrees
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-9.81,0,0])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')

    def test_tilt_and_constant_angular_velocity(self):
        # Pitch forward by 90 degrees, and rotate about local x axis by 1 rad/s
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([1, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-9.81,0,0])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([1,0,0])), '')

    def test_tilt_and_acceleration(self):
        # Pitch forward by 90 degrees, and accelerate towards global x axis by 1 m/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([1, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-9.81,0,1])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')


class SensorTranslationTestCase(unittest.TestCase):
    def setUp(self):
        gravity_vector = np.array([0, 0 , -9.81])
        
        # The IMU is sitting 0.1m above the quadrotor origin 
        self.r = 0.1
        R_BS = np.identity(3)
        p_BS = np.zeros(3)
        p_BS[2] = self.r
        accelerometer_noise_density = 0.0
        accelerometer_random_walk = 0.0
        gyroscope_noise_density = 0.0
        gyroscope_random_walk = 0.0
        self.imu = Imu(R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, gravity_vector)

    def test_tilt_and_constant_angular_velocity_and_angular_acceleration(self):
        # Pitch forward by 90 degrees, and rotate about local x axis by 1 rad/s, with angular acceleration about local y axis by 0.3 rad/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([1, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0.3, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        acceleration_from_angular_vel = self.r * state['w'][0]**2  # r * omega^2
        acceleration_from_angular_aceel = self.r * acceleration['wdot'][1]
        # minus sign in acceleration_from_angular_vel because the mass inside the imu is pulled away from the rotation axis, which gives a measured direction toward the axis
        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-9.81 + acceleration_from_angular_aceel, 0, -acceleration_from_angular_vel])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, state['w']), '')


    def test_tilt_and_constant_angular_acceleration(self):
        # Pitch forward by 90 degrees, and rotate with angular acceleration about local x axis by 1 rad/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([1, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        acceleration_from_angular_aceel = self.r * acceleration['wdot'][0]
        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-9.81, -acceleration_from_angular_aceel, 0])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, state['w']), '')


class SensorRotationTestCase(unittest.TestCase):
    def setUp(self):
        gravity_vector = np.array([0, 0 , -9.81])
        
        # We get IMU frame by pitching body frame backward by 90 degrees 
        R_BS = np.zeros((3,3))
        R_BS[1,1] = 1
        R_BS[2,0] = 1
        R_BS[0,2] = -1
        p_BS = np.zeros(3)
        accelerometer_noise_density = 0.0
        accelerometer_random_walk = 0.0
        gyroscope_noise_density = 0.0
        gyroscope_random_walk = 0.0
        self.imu = Imu(R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, gravity_vector)

    def test_stationary(self):
        # Completley stationary.
        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array([0, 0, 0, 1]),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([9.81,0,0])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')

    def test_stationary_tilt(self):
        # Stationary but pitch forward by 90 degrees
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([0,0,9.81])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')

    def test_tilt_and_constant_angular_velocity(self):
        # Pitch forward by 90 degrees, and rotate about local x axis by 1 rad/s
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([1, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([0,0,9.81])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,-1])), '')

    def test_tilt_and_acceleration(self):
        # Pitch forward by 90 degrees, and accelerate towards global x axis by 1 m/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([1, 0, 0]),
        'wdot': np.array([0, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        self.assertTrue(np.allclose(accelerometer_measurement, np.array([1,0,9.81])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')


class SensorTranslationAndRotationTestCase(unittest.TestCase):
    def setUp(self):
        gravity_vector = np.array([0, 0 , -9.81])
        
        # We get IMU frame by pitching body frame backward by 90 degrees 
        # The IMU is sitting 0.1m above the quadrotor origin in quadrotor's frame
        R_BS = np.zeros((3,3))
        R_BS[1,1] = 1
        R_BS[2,0] = 1
        R_BS[0,2] = -1
        self.r = 0.1
        p_BS = np.zeros(3)
        p_BS[2] = self.r
        accelerometer_noise_density = 0.0
        accelerometer_random_walk = 0.0
        gyroscope_noise_density = 0.0
        gyroscope_random_walk = 0.0
        self.imu = Imu(R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, gravity_vector)

    def test_tilt_and_constant_angular_velocity_and_angular_acceleration(self):
        # Pitch forward by 90 degrees, and rotate about local x axis by 1 rad/s, with angular acceleration about local y axis by 0.3 rad/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([1, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([0, 0.3, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        acceleration_from_angular_vel = self.r * state['w'][0]**2  # r * omega^2
        acceleration_from_angular_aceel = self.r * acceleration['wdot'][1]
        # minus sign in acceleration_from_angular_vel because the mass inside the imu is pulled away from the rotation axis, which gives a measured direction toward the axis
        self.assertTrue(np.allclose(accelerometer_measurement, np.array([-acceleration_from_angular_vel, 0, 9.81 - acceleration_from_angular_aceel])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,-state['w'][0]])), '')


    def test_tilt_and_constant_angular_acceleration(self):
        # Pitch forward by 90 degrees, and rotate with angular acceleration about local x axis by 1 rad/s^2
        theta = math.pi / 2
        axis = np.array([0, 1, 0])

        state = {'x': np.array([0, 0, 0]),
        'v': np.array([0, 0, 0]),
        'q': np.array(np.hstack([math.sin(theta/2) * axis, math.cos(theta/2)])),
        'w': np.array([0, 0, 0])}

        acceleration = {'vdot': np.array([0, 0, 0]),
        'wdot': np.array([1, 0, 0])}

        accelerometer_measurement, gyroscope_measurement = self.imu.measurement(state, acceleration, False)

        acceleration_from_angular_aceel = self.r * acceleration['wdot'][0]
        self.assertTrue(np.allclose(accelerometer_measurement, np.array([0, -acceleration_from_angular_aceel, 9.81])), '')
        self.assertTrue(np.allclose(gyroscope_measurement, np.array([0,0,0])), '')


if __name__ == '__main__':
    unittest.main()


