import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse import SparseWarning
from scipy.sparse import lil_matrix

from .graph_search import graph_search

def path_prune(points):
    pruned_points = list(points)
    count = 0
    while count != len(pruned_points) - 2:
        if count > len(pruned_points) - 2:
            break
        direction = np.cross(pruned_points[count] - pruned_points[count + 1], pruned_points[count + 1] - pruned_points[count + 2])    #check the direction vector
        dist = np.linalg.norm(pruned_points[count] - pruned_points[count + 1])     #check the distance between next two points
        if np.linalg.norm(direction) == 0:
            del pruned_points[count + 1]
            count -= 1
        elif dist > 0.01:
            del pruned_points[count]
        count += 1

    pruned_points = np.delete(pruned_points, 1, 0)

    return pruned_points


class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """
        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.255, 0.25, 0.25])

        self.margin = 0.5
        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        self.points = path_prune(self.path)       # Pruning the points to remove waypoints

        self.v = 6.2

        self.d = np.linalg.norm(self.points[1::]-self.points[0:-1], axis=1).reshape(-1, 1)

    def update(self, t):
        """
        PRIMARY METHOD
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        time = self.d / self.v
        time[0] = 3 * time[0]
        time[-1] = 3 * time[-1]

        coeff = np.sqrt(1.65 / time)
        time = time * coeff

        cummul_time = np.vstack((np.zeros(1), np.cumsum(time, axis=0))).flatten()
        n = self.points.shape[0]
        m = 8 * (n-1)
        B = lil_matrix((m, 3))
        time = time.clip(0.25, np.inf)
        total_time = np.sum(time)

        for i in range(n - 1):
            B[8 * i + 3] = self.points[i]
            B[8 * i + 4] = self.points[i + 1]

        A = lil_matrix((m, m))
        iter = 0
        for ti in time:

            sub = np.array([[0           , 0          , 0          , 0          , 0         , 0        , 0    , 1],
                            [ti**7       , ti**6      , ti**5      , ti**4      , ti**3     , ti**2    , ti   , 1],
                            [7 * ti**6   , 6 * ti**5  , 5 * ti**4  , 4 * ti**3  , 3 * ti**2 , 2 * ti   , 1    , 0],
                            [42 * ti**5  , 30 * ti**4 , 20 * ti**3 , 12 * ti**2 , 6 * ti    , 2        , 0    , 0],
                            [210 * ti**4 , 120 * ti**3, 60 * ti**2 , 24 * ti    , 6         , 0        , 0    , 0],
                            [840 * ti**3 , 360 * ti**2, 120 * ti   , 24         , 0         , 0        , 0    , 0],
                            [2520 * ti**2, 720 * ti   , 120        , 0          , 0         , 0        , 0    , 0],
                            [5040 * ti   , 720        , 0          , 0          , 0         , 0        , 0    , 0]])

            A[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]
            if iter != len(time) - 1:
                A[5 + 8*iter, 14 + 8*iter] = -1
                A[6 + 8*iter, 13 + 8*iter] = -2
                A[7 + 8*iter, 12 + 8*iter] = -6
                A[8 + 8*iter, 11 + 8*iter] = -24
                A[9 + 8*iter, 10 + 8*iter] = -120
                A[10 + 8*iter, 9 + 8*iter] = -720
                A[8*iter + 3:8*iter + 11, 8*iter:8*iter + 8] = sub
            else:
                A[8*iter + 3:8*iter + 11, 8*iter:8*iter + 8] = sub[:5, :]
            iter += 1

        A = A.tocsc()
        C = spsolve(A, B).toarray()

        if t > total_time:
            x = self.points[-1]
        else:
            id = np.where(np.sign(cummul_time - t) > 0)
            id = id[0][0] - 1

            t_arr = np.array([[(t - cummul_time[id])**7, (t - cummul_time[id])**6, (t - cummul_time[id])**5, (t - cummul_time[id])**4, (t - cummul_time[id])**3, (t - cummul_time[id])**2, (t - cummul_time[id]), 1],
                              [7 * (t - cummul_time[id])**6, 6 * (t - cummul_time[id])**5, 5 * (t - cummul_time[id])**4, 4 * (t - cummul_time[id])**3, 3 * (t - cummul_time[id])**2, 2 * (t - cummul_time[id]), 1, 0],
                              [42 * (t - cummul_time[id])**5, 30 * (t - cummul_time[id])**4, 20 * (t - cummul_time[id])**3, 12 * (t - cummul_time[id])**2, 6 * (t - cummul_time[id]), 2, 0, 0],
                              [210 * (t - cummul_time[id])**4, 120 * (t - cummul_time[id])**3, 60 * (t - cummul_time[id])**2, 24 * (t - cummul_time[id]), 6, 0, 0, 0],
                              [840 * (t - cummul_time[id])**3, 360 * (t - cummul_time[id])**2, 120 * (t - cummul_time[id]), 24, 0, 0, 0, 0]])

            x = (t_arr @ C[8*id:8*id+8, :])[0, :]
            x_dot = (t_arr @ C[8 * id:8 * id + 8, :])[1, :]
            x_ddot = (t_arr @ C[8 * id:8 * id + 8, :])[2, :]
            x_dddot = (t_arr @ C[8 * id:8 * id + 8, :])[3, :]
            x_ddddot = (t_arr @ C[8 * id:8 * id + 8, :])[4, :]
        # STUDENT CODE END
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output




