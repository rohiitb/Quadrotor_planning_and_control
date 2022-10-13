from enum import Enum
import functools
import copy
import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

class ExitStatus(Enum):
    """ Exit status values indicate the reason for simulation termination. """
    COMPLETE     = 'Success: End reached.'
    TIMEOUT      = 'Timeout: Simulation end time reached.'
    INF_VALUE    = 'Failure: Your controller returned inf motor speeds.'
    NAN_VALUE    = 'Failure: Your controller returned nan motor speeds.'
    OVER_SPEED   = 'Failure: Your quadrotor is out of control; it is going faster than 100 m/s. The Guinness World Speed Record is 73 m/s.'
    OVER_SPIN    = 'Failure: Your quadrotor is out of control; it is spinning faster than 100 rad/s. The onboard IMU can only measure up to 52 rad/s (3000 deg/s).'
    FLY_AWAY     = 'Failure: Your quadrotor is out of control; it flew away with a position error greater than 20 meters.'

def simulate(initial_state, quadrotor, controller, trajectory, t_final, terminate=None, vio = None, stereo = None):
    """
    Perform a quadrotor simulation and return the numerical results.

    Inputs:
        initial_state, a dict defining the quadrotor initial conditions with keys
            x, position, m, shape=(3,)
            v, linear velocity, m/s, shape=(3,)
            q, quaternion [i,j,k,w], shape=(4,)
            w, angular velocity, rad/s, shape=(3,)
        quadrotor, Quadrotor object
        controller, SE3Control object
        trajectory, Trajectory object
        t_final, maximum duration of simulation, s

        terminate, None, False, or a function of time and state that returns
            ExitStatus. If None (default), terminate when hover is reached at
            the location of trajectory with t=inf. If False, never terminate
            before timeout or error. If a function, terminate when returns not
            None.

    Outputs:
        time, seconds, shape=(N,)
        state, a dict describing the state history with keys
            x, position, m, shape=(N,3)
            v, linear velocity, m/s, shape=(N,3)
            q, quaternion [i,j,k,w], shape=(N,4)
            w, angular velocity, rad/s, shape=(N,3)
        control, a dict describing the command input history with keys
            cmd_motor_speeds, motor speeds, rad/s, shape=(N,4)
            cmd_q, commanded orientation (not used by simulator), quaternion [i,j,k,w], shape=(N,4)
            cmd_w, commanded angular velocity (not used by simulator), rad/s, shape=(N,3)
        flat, a dict describing the desired flat outputs from the trajectory with keys
            x,        position, m
            x_dot,    velocity, m/s
            x_ddot,   acceleration, m/s**2
            x_dddot,  jerk, m/s**3
            x_ddddot, snap, m/s**4
            yaw,      yaw angle, rad
            yaw_dot,  yaw rate, rad/s
        exit_status, an ExitStatus enum indicating the reason for termination.
    """

    # Coerce entries of initial state into numpy arrays, if they are not already.
    initial_state = {k: np.array(v) for k, v in initial_state.items()}

    if terminate is None:    # Default exit. Terminate at final position of trajectory.
        if vio is None:
            normal_exit = traj_end_exit(initial_state, trajectory, using_vio = False)
        else:
            normal_exit = traj_end_exit(initial_state, trajectory, using_vio = True)
    elif terminate is False: # Never exit before timeout.
        normal_exit = lambda t, s: None
    else:                    # Custom exit.
        normal_exit = terminate

    t_step = 1/500 # in seconds, determines control loop frequency
    if vio is not None:
        t_step = 1/vio.sampling_rate
        print('Running VIO, reducing simulator to',1/t_step,' Hz!!!')
    time    = [0]
    state   = [copy.deepcopy(initial_state)]
    images_feature = []
    imu_measurements = []
    vio_state = [copy.deepcopy(initial_state)]
    flat    = [sanitize_trajectory_dic(trajectory.update(time[-1]))]
    control = [sanitize_control_dic(controller.update(time[-1], state[-1], flat[-1]))]

    exit_status = None

    # Initialize VIO 
    if vio is not None and not vio.initialized:
        state_dot = quadrotor.statedot(state[0], control[0]['cmd_motor_speeds'], t_step)
        vio.initialize(state[0], state_dot, time[0])
    if vio is not None and not stereo.initialized:
        stereo.sample_features()

    while True:
        exit_status = exit_status or safety_exit(state[-1], flat[-1], control[-1])
        exit_status = exit_status or normal_exit(time[-1], state[-1])
        exit_status = exit_status or time_exit(time[-1], t_final)
        if exit_status:
            break
        time.append(time[-1] + t_step)
        state.append(quadrotor.step(state[-1], control[-1]['cmd_motor_speeds'], t_step))
        flat.append(sanitize_trajectory_dic(trajectory.update(time[-1])))
        if vio is not None:
            state_dot = quadrotor.statedot(state[-1], control[-1]['cmd_motor_speeds'], t_step)
            state_estimated, image_feature, imu_measurement = vio.step(state[-1], state_dot, time[-1], stereo)
            if image_feature is not None:
                # meaning that the camera update is triggered
                images_feature.append(image_feature.T)
            imu_measurements.append(imu_measurement)
            vio_state.append(state_estimated)
            control.append(sanitize_control_dic(controller.update(time[-1], vio_state[-1], flat[-1])))
        else:
            control.append(sanitize_control_dic(controller.update(time[-1], state[-1], flat[-1])))


    time    = np.array(time, dtype=float)    
    state   = merge_dicts(state)
    if vio is not None:
        vio_state  = merge_dicts(vio_state)
    control = merge_dicts(control)
    flat    = merge_dicts(flat)

    if vio is not None:
        return (time, state, vio_state, control, flat, exit_status, imu_measurements)
    else:
        return (time, state, control, flat, exit_status)


def merge_dicts(dicts_in):
    """
    Concatenates contents of a list of N state dicts into a single dict by
    prepending a new dimension of size N. This is more convenient for plotting
    and analysis. Requires dicts to have consistent keys and have values that
    are numpy arrays.
    """
    dict_out = {}
    for k in dicts_in[0].keys():
        dict_out[k] = []
        for d in dicts_in:
            dict_out[k].append(d[k])
        dict_out[k] = np.array(dict_out[k])
    return dict_out

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot


def traj_end_exit(initial_state, trajectory, using_vio = False):
    """
    Returns a exit function. The exit function returns an exit status message if
    the quadrotor is near hover at the end of the provided trajectory. If the
    initial state is already at the end of the trajectory, the simulation will
    run for at least one second before testing again.
    """

    xf = trajectory.update(np.inf)['x']
    yawf = trajectory.update(np.inf)['yaw']
    rotf = Rotation.from_rotvec(yawf * np.array([0, 0, 1])) # create rotation object that describes yaw
    if np.array_equal(initial_state['x'], xf):
        min_time = 1.0
    else:
        min_time = 0

    def exit_fn(time, state):
        cur_attitude = Rotation.from_quat(state['q'])
        err_attitude = rotf * cur_attitude.inv() # Rotation between current and final
        angle = norm(err_attitude.as_rotvec()) # angle in radians from vertical
        # Success is reaching near-zero speed with near-zero position error.
        if using_vio:
            # set larger threshold for VIO due to noisy measurements
            if time >= min_time and norm(state['x'] - xf) < 1 and norm(state['v']) <= 1 and angle <= 1:
                return ExitStatus.COMPLETE
        else:
            if time >= min_time and norm(state['x'] - xf) < 0.02 and norm(state['v']) <= 0.03 and angle <= 0.02:
                return ExitStatus.COMPLETE
        return None
    return exit_fn

def time_exit(time, t_final):
    """
    Return exit status if the time exceeds t_final, otherwise None.
    """
    if time >= t_final:
        return ExitStatus.TIMEOUT
    return None

def safety_exit(state, flat, control):
    """
    Return exit status if any safety condition is violated, otherwise None.
    """
    if np.any(np.isinf(control['cmd_motor_speeds'])):
        return ExitStatus.INF_VALUE
    if np.any(np.isnan(control['cmd_motor_speeds'])):
        return ExitStatus.NAN_VALUE
    if np.any(np.abs(state['v']) > 100):
        return ExitStatus.OVER_SPEED
    if np.any(np.abs(state['w']) > 100):
        return ExitStatus.OVER_SPIN
    if np.any(np.abs(state['x'] - flat['x']) > 20):
        return ExitStatus.FLY_AWAY
    return None

def sanitize_control_dic(control_dic):
    """
    Return a sanitized version of the control dictionary where all of the elements are np arrays
    """
    control_dic['cmd_motor_speeds'] = np.asarray(control_dic['cmd_motor_speeds'], np.float).ravel()
    control_dic['cmd_moment'] = np.asarray(control_dic['cmd_moment'], np.float).ravel()
    control_dic['cmd_q'] = np.asarray(control_dic['cmd_q'], np.float).ravel()
    return control_dic

def sanitize_trajectory_dic(trajectory_dic):
    """
    Return a sanitized version of the trajectory dictionary where all of the elements are np arrays
    """
    trajectory_dic['x'] = np.asarray(trajectory_dic['x'], np.float).ravel()
    trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], np.float).ravel()
    trajectory_dic['x_ddot'] = np.asarray(trajectory_dic['x_ddot'], np.float).ravel()
    trajectory_dic['x_dddot'] = np.asarray(trajectory_dic['x_dddot'], np.float).ravel()
    trajectory_dic['x_ddddot'] = np.asarray(trajectory_dic['x_ddddot'], np.float).ravel()

    return trajectory_dic

class Quadrotor(object):
    """
    Quadrotor forward dynamics model.
    """
    def __init__(self, quad_params):
        """
        Initialize quadrotor physical parameters.
        """

        # Read physical parameters out of quad_params.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # Additional constants.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [ k, -k,  k, -k]])
        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

    def statedot(self, state, cmd_rotor_speeds, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        # The true motor speeds can not fall below min and max speeds.
        rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Compute individual rotor thrusts and net thrust and net moment.
        rotor_thrusts = self.k_thrust * rotor_speeds**2
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]
        M = TM[1:4]

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)
        s = Quadrotor._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, cmd_rotor_speeds, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        # The true motor speeds can not fall below min and max speeds.
        rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Compute individual rotor thrusts and net thrust and net moment.
        rotor_thrusts = self.k_thrust * rotor_speeds**2
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]
        M = TM[1:4]

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)
        s = Quadrotor._pack_state(state)
        
        # Option 1 - RK45 integration
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s = sol['y'][:,-1]
        # Option 2 - Euler integration
        # s = s + s_dot_fn(0, s) * t_step  # first argument doesn't matter. It's time invariant model

        state = Quadrotor._unpack_state(s)

        # Re-normalize unit quaternion.
        state['q'] = state['q'] / norm(state['q'])

        return state

    def _s_dot_fn(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Quadrotor._unpack_state(s)

        # Position derivative.
        x_dot = state['v']

        # Velocity derivative.
        F = u1 * Quadrotor.rotate_k(state['q'])
        v_dot = (self.weight + F) / self.mass

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Angular velocity derivative.
        omega = state['w']
        omega_hat = Quadrotor.hat_map(omega)
        w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))

        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot

        return s_dot

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[    0, -s[2],  s[1]],
                         [ s[2],     0, -s[0]],
                         [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((13,))
        s[0:3]   = state['x']
        s[3:6]   = state['v']
        s[6:10]  = state['q']
        s[10:13] = state['w']
        return s

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13]}
        return state
