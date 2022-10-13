#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    R = Rotation.as_matrix(q)

    new_p = p + v * dt + (0.5 * dt ** 2) * (np.dot(R, a_m - a_b) + g)

    new_v = v + (np.dot(R, a_m - a_b) + g) * dt

    delta_R = ((w_m - w_b) * dt)
    delta_q = Rotation.from_rotvec(delta_R.reshape(-1))
    delta_mat = Rotation.as_matrix(delta_q)
    rot = R @ delta_mat
    new_q = Rotation.from_matrix(rot)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = Rotation.as_matrix(q)
    w_rot = (dt * (w_m - w_b)).reshape(-1)
    R_T = Rotation.from_rotvec(w_rot)
    R_T_T = Rotation.as_matrix(R_T)
    a_m = a_m.reshape(-1)
    a_b = a_b.reshape(-1)

    R_a = np.array([[0, -(a_m[2] - a_b[2]), (a_m[1] - a_b[1])],
                    [(a_m[2] - a_b[2]), 0, -(a_m[0] - a_b[0])],
                    [-(a_m[1] - a_b[1]), (a_m[0] - a_b[0]), 0]])

    F_x = np.zeros((18, 18))
    F_x[:3, :3] = np.eye(3)
    F_x[:3, 3:6] = dt * np.eye(3)
    F_x[3:6, 3:6] = np.eye(3)
    F_x[3:6, 6:9] = (R @ R_a) * (-dt)
    F_x[3:6, 9:12] = R * (-dt)
    F_x[3:6, 15:18] = dt * np.eye(3)
    F_x[6:9, 6:9] = R_T_T.T
    F_x[6:9, 12:15] = -dt * np.eye(3)
    F_x[9:12, 9:12] = np.eye(3)
    F_x[12:15, 12:15] = np.eye(3)
    F_x[15:18, 15:18] = np.eye(3)

    F_i = np.zeros((18, 12))
    F_i[3:6, :3] = np.eye(3)
    F_i[6:9, 3:6] = np.eye(3)
    F_i[9:12, 6:9] = np.eye(3)
    F_i[12:15, 9:12] = np.eye(3)

    V_i = (accelerometer_noise_density ** 2) * (dt ** 2) * np.eye(3)
    theta_i = (gyroscope_noise_density ** 2) * (dt ** 2) * np.eye(3)
    A_i = (accelerometer_random_walk ** 2) * dt * np.eye(3)
    omega_i = (gyroscope_random_walk ** 2) * dt * np.eye(3)

    Q_i = np.zeros((12, 12))
    Q_i[:3, :3] = V_i
    Q_i[3:6, 3:6] = theta_i
    Q_i[6:9, 6:9] = A_i
    Q_i[9:12, 9:12] = omega_i

    P = (F_x @ error_state_covariance @ F_x.T) + (F_i @ Q_i @ F_i.T)

    # return an 18x18 covariance matrix
    return P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    innovation = np.zeros((2, 1))
    R = Rotation.as_matrix(q)
    Pc = R.T @ (Pw - p)
    Pc_normalized = (Pc / Pc[2]).reshape(-1, 1)
    innovation = uv - Pc_normalized[:2]

    if np.linalg.norm(innovation) < error_threshold:
        P_c_0 = R.T @ (Pw - p)
        d_Pc_d_theta = np.array([[0, float(-P_c_0[2]), float(P_c_0[1])],
                                 [float(P_c_0[2]), 0, float(-P_c_0[0])],
                                 [float(-P_c_0[1]), float(P_c_0[0]), 0]])

        d_Pc_d_p = -R.T

        d_zt_d_Pc = (1 / float(Pc[2])) * np.array([[1, 0, float(-Pc_normalized[0])],
                                                   [0, 1, float(-Pc_normalized[1])]])

        d_zt_d_theta = d_zt_d_Pc @ d_Pc_d_theta
        d_zt_d_p = d_zt_d_Pc @ d_Pc_d_p

        H = np.zeros((2, 18))
        H[:2, :3] = d_zt_d_p
        H[:2, 6:9] = d_zt_d_theta

        K = error_state_covariance @ H.T @ np.linalg.inv((H @ error_state_covariance @ H.T) + Q)
        delta_x = K @ innovation
        error_state_covariance = ((np.eye(18) - (K @ H)) @ error_state_covariance @ (np.eye(18) - (K @ H)).T) + (
                K @ Q @ K.T)

        p = p + delta_x[:3]
        v = v + delta_x[3:6]
        a_b = a_b + delta_x[9:12]
        w_b = w_b + delta_x[12:15]
        g = g + delta_x[15:18]
        R_new = Rotation.from_rotvec(delta_x[6:9].reshape(-1)).as_matrix()
        new_q = R @ R_new
        q = Rotation.from_matrix(new_q)

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
