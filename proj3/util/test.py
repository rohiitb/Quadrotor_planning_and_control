import contextlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import timeout_decorator
import unittest

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.spatial.distance import cdist

from flightsim.crazyflie_params import quad_params
from flightsim.numpy_encoding import NumpyJSONEncoder
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World
from flightsim.sensors.vio_utils import Vio
from flightsim.sensors.stereo_utils import StereoUtils

def calculate_metrics(results, world, goal):
    """
    Return metrics based on results.
    """

    robot_radius = 0.25
    collision_pts = world.path_collisions(results['state']['x'], robot_radius)
    if collision_pts.size > 0:
        no_collision = False
        collision_point = collision_pts[0]
    else:
        no_collision = True
        collision_point = None


    metrics = {}
    metrics['stopped_at_goal'] = bool((results['exit'] == ExitStatus.COMPLETE) and np.linalg.norm(results['state']['x'][-1] - goal) <= 2.00)
    metrics['no_collision'] = no_collision
    metrics['flight_time'] = float(round(results['time'][-1], 2))
    metrics['flight_distance'] = float(round(np.sum(np.linalg.norm(np.diff(results['state']['x'], axis=0),axis=1)), 2))
    metrics['planning_time'] = results['planning_time']
    metrics['collision_point'] = collision_point

    # Details about why the simulation ended (success, failure, timeout).
    metrics['sim_exit'] = results['exit'].value

    return metrics


def test_mission(traj_cls, se3_control_cls, world, start, goal):
    """
    Test the provided graph_search function against a world, start, and goal.
    Return the simulation results and the performance metrics.
    """

    # Student code to test.
    start_time = time.time()
    my_traj = traj_cls(world, start, goal)
    planning_time = round(time.time() - start_time, 2)
    my_se3_control = se3_control_cls(quad_params)

    # Simulation options.
    quadrotor = Quadrotor(quad_params)
    t_final = 150

    # Run simulation and collect results.
    initial_state = {'x': start,
                     'v': (0, 0, 0),
                     'q': (0, 0, 0, 1), # [i,j,k,w]
                     'w': (0, 0, 0)}
    vio = Vio()


    # maximum number of features considered for VIO, increasing it will make VIO more robust, but the less efficient
    max_num_features = 150
    # feature sample resolution (in meter), increasing it will make VIO more efficient, but the less robust
    sample_resolution = 0.75
    visualize_stereo_features = False
    stereo = StereoUtils(world, vio.camera_matrix, sample_resolution = sample_resolution, visualization=visualize_stereo_features, max_num_features=max_num_features)
    (sim_time, state, est_state, control, flat, exit, imu_measurements) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_traj,
                                              t_final, stereo=stereo, vio=vio)
    results = {'time':sim_time, 'state':state, 'control':control, 'flat':flat, 'exit':exit, 'planning_time':planning_time}

    # Evaluate metrics.
    metrics = calculate_metrics(results, world, goal)

    return results, metrics


def plot_mission(world, start, goal, results, metrics, test_name):
    """
    Return a figure showing path through trees along with start and end.
    """

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.projections import register_projection
    import matplotlib.pyplot as plt

    from flightsim.axes3ds import Axes3Ds

    # 3D Paths
    flight_fig = plt.figure('3D Path')
    ax = Axes3Ds(flight_fig)
    world.draw(ax)
    ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
    ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
    world.draw_line(ax, results['flat']['x'], color='black', linewidth=2)
    world.draw_points(ax, results['state']['x'], color='blue', markersize=4)
    if not metrics['no_collision']:
        ax.plot([metrics['collision_point'][0]], [metrics['collision_point'][1]], [metrics['collision_point'][2]], 'rx', markersize=36, markeredgewidth=4)
    ax.legend(handles=[
        Line2D([], [], color='black', linewidth=2, label='Trajectory'),
        Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
        loc='upper right')
    ax.set_title("Path through {test_name}", loc='left')

    # # Visualize the original dense path from A*, your sparse waypoints, and the
    # # smooth trajectory.
    path_fig = plt.figure('A* Path, Waypoints, and Trajectory')
    ax = Axes3Ds(path_fig)
    world.draw(ax)
    ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
    ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
    # if results['path'] is not None:
    #     world.draw_line(ax, results['path'], color='red', linewidth=1)
    # if results['points'] is not None:
    #     world.draw_points(ax, results['points'], color='purple', markersize=8)
    # t = np.linspace(0, results['time'][-1], num=100)
    # x = np.zeros((t.size,3))
    # for i in range(t.size):
    #     flat = my_world_traj.update(t[i])
    #     x[i,:] = flat['x']
    # world.draw_line(ax, results['flat']['x'], color='black', linewidth=2)
    # ax.legend(handles=[
    #     Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    #     Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
    #     Line2D([], [], color='black', linewidth=2, label='Trajectory')],
    #     loc='upper right')

    return [flight_fig]


class TestBase(unittest.TestCase):

    traj_cls = None
    se3_control_cls = None

    longMessage = False
    outpath = Path(inspect.getsourcefile(test_mission)).resolve().parent.parent / 'data_out'
    outpath.mkdir(parents=True, exist_ok=True)

    test_names = []

    def helper_test(self, test_name, world, start, goal, std_target):
        """
        Test student's SE3Control and WorldTraj against provided world.
        Run simulation, save metrics to file, save result plots to file.
        """
        with contextlib.redirect_stdout(std_target):  # Context gobbles stdout.
            result_file = self.outpath / ('result_' + test_name + '.json')
            try: # gracefully handle timeout exceptions
                (results, metrics) = test_mission(self.traj_cls,
                                                  self.se3_control_cls,
                                                  world,
                                                  start,
                                                  goal)
                with open(result_file, 'w') as f:
                    f.write(json.dumps(metrics, cls=NumpyJSONEncoder, indent=4))
                    figs = plot_mission(world, start, goal, results, metrics, test_name)
                    # Save all figures to file
                    with PdfPages(self.outpath / ('result_' + test_name + '.pdf')) as pdf:
                        for fig in figs:
                            pdf.savefig(fig)
            except timeout_decorator.timeout_decorator.TimeoutError as err:
                with open(result_file, 'w') as f:
                    output = {'test_name': test_name, 'error': err.value}
                    f.write(json.dumps(output, cls=NumpyJSONEncoder, indent=4))

    @classmethod
    def set_target(cls, module_name):
        """
        Set the target module to test, and load required classes or functions.
        """
        cls.traj_cls = importlib.import_module(module_name + '.world_traj').WorldTraj
        cls.se3_control_cls = importlib.import_module(module_name + '.se3_control').SE3Control

    @classmethod
    def load_tests(cls, files, *, enable_timeouts=False, redirect_stdout=True):
        """
        Add one test for each input file. For each input file named
        "test_XXX.json" creates a new test member function that will generate
        output files "result_XXX.json" and "result_XXX.pdf".
        """
        std_target = None if redirect_stdout else sys.stdout
        for file in files:
            if file.stem.startswith('test_'):
                test_name = file.stem[5:]
                cls.test_names.append(test_name)
                world=World.from_file(file)

                # Timeouts must be enabled.
                timeout = None
                if enable_timeouts:
                    timeout = 180

                # Dynamically add member function for this test.
                @timeout_decorator.timeout(timeout, exception_message="Test reached time limit of {} seconds".format(timeout))
                def fn(self, test_name=test_name,
                       world=world,
                       start=world.world['start'],
                       goal=world.world['goal']):
                    self.helper_test(test_name, world, start, goal, std_target)
                setattr(cls, 'test_' + test_name, fn)
                # Remove any pre-existing output files for this test.
                # TODO: The 'missing_ok' argument becomes available in Python
                # 3.8, at which time contextlib is no longer needed.
                with contextlib.suppress(FileNotFoundError):
                    (cls.outpath / ('result_' + test_name + '.json')).unlink()
                with contextlib.suppress(FileNotFoundError):
                    (cls.outpath / ('result_' + test_name + '.pdf')).unlink()

    @classmethod
    def collect_results(cls):
        results = []
        for name in cls.test_names:
            p = cls.outpath / ('result_' + name + '.json')
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                    data['test_name'] = name
                    results.append(data)
            else:
                results.append({'test_name': name})
        return results

    @classmethod
    def print_results(cls):
        results = cls.collect_results()
        for r in results:
            print()
            if len(r.keys()) > 2:
                passed = r['stopped_at_goal'] and r['no_collision']
                print('{} {}'.format('pass' if passed else 'FAIL', r['test_name']))
                print('    stopped at goal:        {}'.format('pass' if r['stopped_at_goal'] else 'FAIL'))
                print('    no collision:           {}'.format('pass' if r['no_collision'] else 'FAIL'))
                print('    flight time, seconds:   {}'.format(r['flight_time']))
                print('    flight dist, meters:    {}'.format(r['flight_distance']))
                print('    planning time, seconds: {}'.format(r['planning_time']))
                print('    exit message:           {}'.format(r['sim_exit']))
            elif 'error' in r.keys():
                print("FAIL {name}\n"
                      "    {error}".format(name=r['test_name'], error=r['error']))
            else:
                print("FAIL {name}\n"
                      "    Test failed with no results. Review error log.".format(
                    name=r['test_name']))


if __name__ == '__main__':
    """
    Run a test for each "test_*.json" file in this directory. You can add new
    tests by copying and editing these files.
    """
    import argparse
    np.random.seed(0)

    # All arguments are optional, and are not needed to test the student solution.
    default_target = 'proj3.code'
    parser = argparse.ArgumentParser(description='Evaluate one assignment solution.')
    parser.add_argument('--target', default=default_target, type=str,
                        help=f"Run on the code module of this name. Default is {default_target}")
    parser.add_argument('--stdout', action='store_true',
                        help="Allow printing to stdout from inside unittest.")
    p = parser.parse_args()

    if p.stdout:
        print('\n*** WARNING: ENABLED PRINTING TO STDOUT FROM INSIDE UNITTEST ***\n')

    # Set target code module to test.
    if p.target != default_target:
        print(f'\n*** WARNING: RUNNING IN DEBUG MODE USING MODULE {p.target}) ***\n')
    TestBase.set_target(module_name=p.target)

    # Collect tests distributed to students.
    path = Path(inspect.getsourcefile(TestBase)).parent.resolve()
    test_files_local = list(Path(path).glob('test_*.json'))
    # Concatenate full list of tests.
    all_test_files = test_files_local
    TestBase.load_tests(all_test_files, redirect_stdout=not p.stdout)

    # Run tests, results saved in data_out.
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBase)
    unittest.TextTestRunner(verbosity=2).run(suite)

    # Collect results for display.
    TestBase.print_results()
