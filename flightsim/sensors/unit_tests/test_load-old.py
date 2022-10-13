from pathlib import Path
from pyexpat import features
from flightsim.world import World
import inspect
import numpy as np
import json
from flightsim.axes3ds import Axes3Ds
import matplotlib.pyplot as plt
from flightsim.numpy_encoding import NumpyJSONEncoder, to_ndarray
from proj3.code.stereo_utils import StereoUtils
from flightsim.simulate import Quadrotor, simulate, ExitStatus

# Load the test example.
filename = 'test.json'
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
all_blocks = []
for b in world.world.get('blocks', []):
    all_blocks.append(b['extents'])

all_blocks_coord = []
for block in all_blocks:  
    all_blocks_coord.append([np.array([block[0], block[2], block[4]]), np.array([block[1], block[3], block[5]])])

stereo_utils = StereoUtils(world, np.identity(3), np.zeros(3), 0, sample_resolution = 1)
stereo_utils.sample_features()
features = stereo_utils.feature_coords

# odom_pos = np.array([0,0,0])
# orientation = np.array([0,0,0,0])
# valid_features_2d, valid_fatures_3d, H_world_cam1 = check_feature_valid(features, all_blocks_coord, odom_pos, orientation)
#
# camera_origin = np.linalg.pinv(H_world_cam1) @ np.array([0,0,0,1])

fig = plt.figure('A* Path, Waypoints, and Trajectory')
ax = Axes3Ds(fig)
ax.scatter(features[:,0], features[:,1], features[:,2], c='black', s = 1)
# ax.scatter(valid_fatures_3d[:,0], valid_fatures_3d[:,1], valid_fatures_3d[:,2], c='red', s = 20)
# ax.scatter(camera_origin[0], camera_origin[1], camera_origin[2], c='orange', s=100)

world.draw(ax)
plt.show()
print(features)
