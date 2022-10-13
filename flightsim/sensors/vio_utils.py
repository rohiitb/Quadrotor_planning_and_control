#%% Imports

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import yaml
from flightsim.sensors.stereo_utils import StereoUtils
from numpy.linalg import norm
from flightsim.sensors.imu import Imu

#######################################################################
from proj3.code.vio import *
################################################################################################################################################################


class Vio():
    def __init__(self):
        self.sampling_rate = 200 #10000 #200

        # Set up IMU
        # Noise parameters from https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
        # IMU configuration
        # IMU0:
        #   Model: calibrated
        #   Update rate: 400.0
        #   Accelerometer:
        #     Noise density: 0.001
        #     Noise density (discrete): 0.02
        #     Random walk: 0.0001
        #   Gyroscope:
        #     Noise density: 0.0001
        #     Noise density (discrete): 0.002
        #     Random walk: 1e-05
        accelerometer_noise_density = 0.1
        accelerometer_random_walk = 0.01
        gyroscope_noise_density = 0.01
        gyroscope_random_walk = 0.001

        # Extract rotation that transforms IMU to left camera frame
        # body to left camera rotation
        self.R_body2sensor =  Rotation.from_euler('zyx', [0, 0, 100], degrees=True).as_matrix() #np.identity(3) #dataset.stereo_calibration.tr_base_left[0:3, 0:3].T


        R_BS = self.R_body2sensor.T #  sensor to body

        p_BS = np.zeros(3)
        self.gravity_vector = np.array([0, 0 , -9.81])
        self.imu = Imu(R_BS, p_BS, accelerometer_noise_density, accelerometer_random_walk, gyroscope_noise_density, gyroscope_random_walk, self.gravity_vector)
        self.accelerometer_noise_density, self.accelerometer_random_walk, self.gyroscope_noise_density, self.gyroscope_random_walk = self.imu.get_noise_parameters()
        # print("imu.get_noise_parameters() = ", self.imu.get_noise_parameters())

        self.prev_uv = None
        self.prev_index_of_valid_features = None
        self.prev_depth = None


        # dataset rectified camera matrix:
        self.camera_matrix = np.array([[456.715, 0.00, 422.191, 0],
                                        [0.00, 456.715, 196.347, 0],
                                        [0.00, 0.00, 1.00, 0]])
        self.focal_length = self.camera_matrix[0, 0]
        self.error_threshold = (10 / self.focal_length)
        # self.depth_std_dev = 0.0001 # in meters
        image_coord_std_dev = self.error_threshold * 0.1
        self.image_u_std_dev =  image_coord_std_dev# in normalized coordinate (X/Z in cam frame)
        self.image_v_std_dev = image_coord_std_dev# in normalized coordinate (X/Z in cam frame)
        self.image_measurement_covariance = (image_coord_std_dev ** 2) * np.eye(2)
        self.initialized = False


    def initialize(self, gt_state, gt_acceleration, sim_time):
        '''
        This function initializes nominal_state of vio and declare a few camera's variables.
            nominal_state:
                p, linear position, shape=(3,1)
                v, linear velocity, shape=(3,1)
                q, rotation object (not quaternion!)
                a_b, accelerometer bias, shape=(3,1)
                w_b, gyroscope bias, shape=(3,1)
                g, gravitational acceleration, shape=(3,1)
        Inputs:
            gt_state: a dict defining the quadrotor initial conditions with keys
                x, position, m, shape=(3,)
                v, linear velocity, m/s, shape=(3,)
                q, quaternion [i,j,k,w], shape=(4,)
                w, angular velocity, rad/s, shape=(3,)
            gt_acceleration, a dict describing the acceleration with keys
                vdot, quadrotor's linear acceleration expressed in world frame, m/s^2, shape=(3,)
                wdot, quadrotor's angular acceleration expressed in world frame, rad/s^2, shape=(3,)
            sim_time, current time in simulation

        '''
        if self.initialized == False:
            # Initialize state
            self.p = gt_state['x'].reshape((3,1))
            self.v = gt_state['v'].reshape((3,1))
            # Important: since the simulator uses body-frame state, but the VIO filter uses camera-frame state as nominal state, we need to transform between them during input / output
            self.q = Rotation.from_matrix((Rotation.from_quat(gt_state['q']).as_matrix()) @ self.R_body2sensor.T)
            # a_b and w_b are sensor bias
            self.a_b = np.zeros((3, 1))
            self.w_b = np.zeros((3, 1))

            # We cannot initialize self.g here from IMU measurement, because the init state might not be stationary
            self.g = self.gravity_vector
            self.g = self.g.reshape((3,1))
            # print("in initialization, g = ", self.R_body2sensor @  Rotation.from_quat(gt_state['q']).as_matrix() @ self.imu.measurement(gt_state, gt_acceleration, with_noise=True)[0])
            # print("in initialization, g = ", self.g)

            self.nominal_state = self.p, self.v, self.q, self.a_b, self.w_b, self.g

            # Initialize error state covariance
            self.error_state_covariance = np.diag([0, 0, 0, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0, 0, 0])

            # These variables encode last stereo pose
            self.last_R = self.nominal_state[2].as_matrix()
            self.last_t = self.nominal_state[0]

            self.trace_covariance = []
            self.pose = []
            self.cov = []
            # self.last_timestamp = sim_time # - 1/self.sampling_rate
            self.dt = 1/self.sampling_rate
            self.initialized = True
            self.imu_counter = 1000 # initialize with a large number so that the first iteration will run measurement update

    # %% Main Loop
    def step(self, gt_state, gt_acceleration, sim_time, stereo):
        '''
        This function estimates quadrotor's state for a simulation step
        Inputs:
            gt_state: a dict defining the quadrotor initial conditions with keys
                x, position, m, shape=(3,)
                v, linear velocity, m/s, shape=(3,)
                q, quaternion [i,j,k,w], shape=(4,)
                w, angular velocity (in local frame!), rad/s, shape=(3,)
            gt_acceleration, a dict describing the acceleration with keys
                vdot, quadrotor's linear acceleration expressed in world frame, m/s^2, shape=(3,)
                wdot, quadrotor's angular acceleration expressed in world frame, rad/s^2, shape=(3,)
            sim_time, current time in simulation in seconds
            stereo, ??? TODO
        Outputs:
            estiated_state: a dict defining the quadrotor state with
                x, position, m, shape=(3,)
                v, linear velocity, m/s, shape=(3,)
                q, quaternion [i,j,k,w], shape=(4,)
                w, angular velocity, rad/s, shape=(3,)
        '''
        self.cov.append(self.error_state_covariance)
        self.trace_covariance.append(self.error_state_covariance.trace())
        self.pose.append((self.nominal_state[2], self.nominal_state[0], self.nominal_state[1], self.nominal_state[3].copy()))

        # Extract prevailing a_m and w_m - transform to left camera frame
        linear_acceleration, angular_velocity = self.imu.measurement(gt_state, gt_acceleration,with_noise=True)
        self.w_m = angular_velocity.reshape(3, 1)
        self.a_m = linear_acceleration.reshape(3, 1)

        # Apply IMU updates
        self.error_state_covariance = error_covariance_update(self.nominal_state, self.error_state_covariance, self.w_m, self.a_m, self.dt,
                                                        self.accelerometer_noise_density, self.gyroscope_noise_density,
                                                        self.accelerometer_random_walk, self.gyroscope_random_walk)
        self.nominal_state = nominal_state_update(self.nominal_state, self.w_m, self.a_m, self.dt)

        # run stereo update less frequently than IMU
        self.imu_counter += 1
        uv_new = None
        # vio will be running at (IMU_RATE / vio_stereo_update_interval) hz, IMU rate is set as 200 HZ
        vio_stereo_update_interval = 10
        if self.imu_counter >= vio_stereo_update_interval:
            self.imu_counter = 0
            # TODO (final integration): make sure this is generated as expected
            # u v and d are, respectively normalized u=X/Z=((u1 - cx) / f), v=Y/Z=((v1 - cy) / f), and d=1/Z=(d1 / (f * stereo_baseline)) where d1 is disparity of features in the left camera in the previous and current frame (uvd1 and uvd2), here we modified to get rid of d since we already know feature's position in camera frame
            # uvd1, uvd2 = temporal_match.get_normalized_matches(dataset.rectified_camera_matrix, dataset.stereo_baseline)
            odom_pos = gt_state['x']
            odom_rot = (Rotation.from_quat(gt_state['q'])).as_matrix()
            valid_features_normalized_uv, valid_features_inverse_depth, P1_all, index_of_valid_features = stereo.get_valid_features(odom_pos, odom_rot, self.R_body2sensor)
            if valid_features_normalized_uv.shape[0] == 0:
                print("FEATURE EXTRACTION: NO FEATURE FOUND, SKIPPING THIS UPDATE STEP FOR STEREO CAMERAS!")
            else:
                uv_new = valid_features_normalized_uv
                depth = 1.0 / valid_features_inverse_depth
                # each column is a feature
                uv_new = uv_new.T
                if self.prev_uv is None:
                    # initialize
                    self.prev_uv = uv_new
                    self.prev_index_of_valid_features = index_of_valid_features
                    self.prev_depth = depth
                else:
                    common_features_index, prev_ind_common, ind_common = np.intersect1d(self.prev_index_of_valid_features, index_of_valid_features, return_indices=True)
                    if len(ind_common) == 0:
                        print("FEATURE TRACK: NO VALID FEATURE TRACK FOUND, SKIPPING THIS UPDATE STEP FOR STEREO CAMERAS!")
                    else:
                        # only perform measurement update if there are valid features found!]
                        innovations = np.zeros((2, common_features_index.shape[0]))
                        for i in range(0, common_features_index.shape[0]):
                            # Compute Pw
                            u1_prev, v1_prev = self.prev_uv[:, prev_ind_common[i]]
                            d_prev = self.prev_depth[prev_ind_common[i]]
                            d1 = 1.0 / d_prev

                            if d1 > 0:
                                P1 = np.array([u1_prev / d1, v1_prev / d1, 1 / d1]).reshape(3, 1)

                                Pw = self.last_R @ P1 + self.last_t

                                # Extract uv, simulate noisy measurements of current observation
                                u_noise = np.random.normal(0,self.image_u_std_dev,2)[0]
                                v_noise = np.random.normal(0,self.image_v_std_dev,1)[0]
                                uv = uv_new[:, ind_common[i]].reshape(2, 1)
                                uv[0] += u_noise
                                uv[1] += v_noise

                                self.nominal_state, self.error_state_covariance, inno = measurement_update_step(self.nominal_state,
                                                                                                    self.error_state_covariance,
                                                                                                    uv, Pw, self.error_threshold,
                                                                                                    self.image_measurement_covariance)

                                innovations[:, i] = inno.ravel()

                        self.prev_uv = uv_new
                        self.prev_depth = depth
                        self.prev_index_of_valid_features = index_of_valid_features

                        count = (norm(innovations, axis=0) < self.error_threshold).sum()

                        pixel_error = np.median(abs(innovations), axis=1) * self.focal_length
                        print("{} / {} inlier ratio, x_error {:.4f}, y_error {:.4f}, norm_v {:.4f}".format(count, uv_new.shape[1],
                                                                                                           pixel_error[0],
                                                                                                           pixel_error[1],
                                                                                                           norm(self.nominal_state[
                                                                                                                    1])))

                        # These variables encode last stereo pose
                        self.last_R = self.nominal_state[2].as_matrix()
                        self.last_t = self.nominal_state[0]


        # Important: since the simulator uses body-frame state, but the VIO filter uses camera-frame state as nominal state, we need to transform between them during input / output
        estimated_state_body_frame = {'x': self.nominal_state[0].reshape((3,)),
                'v': self.nominal_state[1].reshape((3,)),
                'q': Rotation.from_matrix(self.nominal_state[2].as_matrix() @ self.R_body2sensor).as_quat().reshape((4,)),
                'w': (self.R_body2sensor.T @ (self.w_m - self.nominal_state[4])).reshape((3,))}
        return estimated_state_body_frame, uv_new, ( self.R_body2sensor.T @ linear_acceleration,  self.R_body2sensor.T @ angular_velocity)




