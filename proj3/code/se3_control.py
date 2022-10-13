import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        x_des = flat_output['x'].reshape(3, 1)
        x_dot_des = flat_output['x_dot'].reshape(3, 1)
        x_ddot_des = flat_output['x_ddot'].reshape(3, 1)
        # x_dddot_des = flat_output['x_dddot']
        # x_ddddot_des = flat_output['x_ddddot']
        psi_des = flat_output['yaw']
        psi_dot_des = flat_output['yaw_dot']

        x = state['x'].reshape(3, 1)
        x_dot = state['v'].reshape(3, 1)
        quat = state['q']
        w = state['w']

        K_p = np.diag(np.array([7.25, 7.25, 20]))
        K_d = np.diag(np.array([4.4, 4.4, 7]))

        K_r = np.diag(np.array([2600, 2600, 150]))
        K_w = np.diag(np.array([130, 130, 80]))

        X_ddot_cmd = x_ddot_des - np.dot(K_d, x_dot - x_dot_des) - np.dot(K_p, x - x_des)

        F_des = (self.mass * X_ddot_cmd) + np.array([[0], [0], [self.mass * self.g]])

        r = Rotation.from_quat(quat)
        R = r.as_matrix()

        b3 = np.dot(R, np.array([[0], [0], [1]]))

        u1 = np.dot(b3.T, F_des)

        b3_des = F_des / np.linalg.norm(F_des)

        a_psi = np.array([[np.cos(psi_des)], [np.sin(psi_des)], [0]])

        b2_des = np.cross(b3_des.reshape(1, 3), a_psi.reshape(1, 3)) / np.linalg.norm(
            np.cross(b3_des.reshape(1, 3), a_psi.reshape(1, 3)))
        b2_des = b2_des.reshape(3, 1)

        R_des = np.concatenate([np.cross(b2_des.reshape(1, 3), b3_des.reshape(1, 3)).reshape(3, 1), b2_des, b3_des],
                               axis=1)

        t = np.dot(R_des.T, R) - np.dot(R.T, R_des)
        e_r = 0.5 * np.array([[t[2, 1]], [t[0, 2]], [t[1, 0]]])
        e_w = w.reshape(3, 1)
        error_mat = -np.dot(K_r, e_r) - np.dot(K_w, e_w)

        u2 = np.dot(self.inertia, error_mat)

        u = np.vstack((u1, u2))

        L = self.arm_length
        A = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                      [0, self.k_thrust * L, 0, -self.k_thrust * L],
                      [-self.k_thrust * L, 0, self.k_thrust * L, 0],
                      [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])

        motor_speed = np.linalg.inv(A) @ u
        F = np.sqrt(np.where(motor_speed < 0, 0, motor_speed))
        cmd_motor_speeds = F.reshape(4, )

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
