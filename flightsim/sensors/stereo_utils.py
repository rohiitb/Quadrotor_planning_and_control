import time

import numpy as np
import matplotlib.pyplot as plt
from flightsim.axes3ds import Axes3Ds
import os
class StereoUtils():
    """
    Simulated stereo camera 
    """

    def __init__(self, world, camera_matrix, sample_resolution = 0.5, visualization = False, max_num_features=75):
        self.Hit = None
        self.visualize = visualization
        self.plot_counter = 0
        self.plot_initialized = False
        self.visualize_interval = 10
        self.visualize_counter = 0
        print('Visualizer for stereo features is set as: ', self.visualize)
        self.feature_coords = None
        # sampling resolution in meters

        self.sample_res = sample_resolution 
        self.all_blocks = []
        self.world = world
        for b in world.world.get('blocks', []):
            self.all_blocks.append(b['extents'])
        self.all_blocks_coord = []
        for block in self.all_blocks:  
            self.all_blocks_coord.append([np.array([block[0], block[2], block[4]]), np.array([block[1], block[3], block[5]])])
        self.walls = world.world['bounds']['extents']

        # distance threshold in meters, features that are farther away than this threshold are not considered
        self.dist_threshold = 5
        # if number of features within distance threshold > max_count, we choose the closest max_count features (bound computation complexity)
        self.max_count = max_num_features
        self.valid_features_uvd_cam1 = None
        self.initialized = False
        self.camera_matrix = camera_matrix

    # Step 1: given world map, generate world 3D point features
    def sample_features(self):
        '''
        List of all blocks, each row should be xmin (0), xmax (1), 
                                            ymin (2), ymax (3), 
                                            zmin (4), zmax (5) for one block
        Sample feature evenly on block surfaces and wall surfaces
        '''
        # Faces:
        # [xmin, ymin, zmin] -> [xmin, ymax, zmax]
        # [xmin, ymin, zmin] -> [xmax, ymin, zmax]
        # [xmax, ymin, zmax] -> [xmin, ymax, zmax]
        # [xmax, ymin, zmin] -> [xmax, ymax, zmax]
        # [xmax, ymax, zmin] -> [xmin, ymax, zmax]
        # [xmin, ymin, zmin] -> [max, ymax, zmin]
        feature_coords = []
        small = 1.e-5
        res = self.sample_res
        all_blocks = self.all_blocks
        self.initialized = True
        for block in all_blocks:
            # generate six faces
            # Face 1:
            f1_x_values = np.array([block[0]])
            f1_y_values = np.arange(block[2], block[3]+small, step=res)
            f1_z_values = np.arange(block[4], block[5]+small, step=res)
            # Face 2:
            f2_x_values = np.arange(block[0], block[1]+small, step=res)
            f2_y_values = np.array([block[2]])
            f2_z_values = np.arange(block[4], block[5]+small, step=res)
            # Face 3:
            f3_x_values = np.arange(block[0], block[1]+small, step=res)
            f3_y_values = np.arange(block[2], block[3]+small, step=res)
            f3_z_values = np.array([block[5]])
            # Face 4:
            f4_x_values = np.array([block[1]])
            f4_y_values = np.arange(block[2], block[3]+small, step=res)
            f4_z_values = np.arange(block[4], block[5]+small, step=res)
            # Face 5:
            f5_x_values = np.arange(block[0], block[1]+small, step=res)
            f5_y_values = np.array([block[3]])
            f5_z_values = np.arange(block[4], block[5]+small, step=res)
            # Face 6:
            f6_x_values = np.arange(block[0], block[1]+small, step=res)
            f6_y_values = np.arange(block[2], block[3]+small, step=res)
            f6_z_values = np.array([block[4]])
            
            x_values = [f1_x_values, f2_x_values, f3_x_values, f4_x_values, f5_x_values, f6_x_values]
            y_values = [f1_y_values, f2_y_values, f3_y_values, f4_y_values, f5_y_values, f6_y_values]
            z_values = [f1_z_values, f2_z_values, f3_z_values, f4_z_values, f5_z_values, f6_z_values]
            for i in range(6):
                face_x_values = x_values[i]
                face_y_values = y_values[i]
                face_z_values = z_values[i]
                for x in face_x_values:
                    for y in face_y_values:
                        for z in face_z_values:
                            feature_coords.append([x, y, z])

        # Walls:
        w1_x_values = np.arange(self.walls[0], self.walls[1]+small, step=res)
        w1_y_values = np.array([self.walls[2]])
        w1_z_values = np.arange(self.walls[4], self.walls[5]+small, step=res)

        w2_x_values = np.array([self.walls[0]])
        w2_y_values = np.arange(self.walls[2], self.walls[3]+small, step=res)
        w2_z_values = np.arange(self.walls[4], self.walls[5]+small, step=res)

        w3_x_values = np.arange(self.walls[0], self.walls[1]+small, step=res)
        w3_y_values = np.array([self.walls[3]])
        w3_z_values = np.arange(self.walls[4], self.walls[5]+small, step=res)

        w4_x_values = np.array([self.walls[1]])
        w4_y_values = np.arange(self.walls[2], self.walls[3]+small, step=res)
        w4_z_values = np.arange(self.walls[4], self.walls[5]+small, step=res)

        # ground
        w5_x_values = np.arange(self.walls[0], self.walls[1]+small, step=res)
        w5_y_values = np.arange(self.walls[2], self.walls[3]+small, step=res)
        w5_z_values = np.array([self.walls[4]+0.1])

        # ceiling
        w6_x_values = np.arange(self.walls[0], self.walls[1]+small, step=res)
        w6_y_values = np.arange(self.walls[2], self.walls[3]+small, step=res)
        w6_z_values = np.array([self.walls[5]-0.1])

        x_values = [w1_x_values, w2_x_values, w3_x_values, w4_x_values, w5_x_values, w6_x_values]
        y_values = [w1_y_values, w2_y_values, w3_y_values, w4_y_values, w5_y_values, w6_y_values]
        z_values = [w1_z_values, w2_z_values, w3_z_values, w4_z_values, w5_z_values, w6_z_values]

        for i in range(len(z_values)):
            face_x_values = x_values[i]
            face_y_values = y_values[i]
            face_z_values = z_values[i]
            for x in face_x_values:
                for y in face_y_values:
                    for z in face_z_values:
                        feature_coords.append([x, y, z])

        self.feature_coords = np.array(feature_coords)


    # Step 2: given world features and odom. For each feature, check the following:
    # (1) distance < distance_threshold, and if number of features > max_count, we choose the closest max_count features
    # (2) project feature to camera, filter out behind-camera features
    # (3) project feature to both cameras, check it is within both camera's field of view
    # (4) take the top (closest) max_count features
    # (5) the line connecting camera and feature does not intersect any obstacles
    def get_valid_features(self, odom_position, odom_rotation, R_body2sensor):
        if self.feature_coords.shape[0] == 0:
            raise Exception("Feature coordinates are not initialized")
        if self.all_blocks_coord == None:
            raise Exception("Blocks are not initialized")
        feature_positions = self.feature_coords

        # step: distance < distance_threshold
        diff = feature_positions-odom_position
        dist_within_thre = (np.linalg.norm(diff, axis = 1) < self.dist_threshold)
        dist_within_thre2 = (np.linalg.norm(diff, axis = 1) > 1.0) # minimum distance

        # maintain an index of all valid features
        index_of_valid_features = np.arange(feature_positions.shape[0])
        remaining_features = feature_positions[dist_within_thre & dist_within_thre2, :]
        index_of_valid_features = index_of_valid_features[dist_within_thre & dist_within_thre2]

        # step: project feature to camera, filter out behind-camera features
        mat = np.zeros((4,4))
        mat[:3,:3] = odom_rotation.T
        mat[:3,3] = -odom_rotation.T @ odom_position
        mat[3,3] = 1
        H_world2body = mat # calculate based on odom_position and odom_orientation
        H_body2cam = np.zeros((4,4))
        H_body2cam[:3,:3] = R_body2sensor
        H_body2cam[3,3] = 1
        H_world2cam1 = H_body2cam @ H_world2body # calculate based on odom_position and odom_orientation

        # We get rid of cam2 completely in this code since we only need cam1's u v disparity
        features_homo_coord = np.column_stack((remaining_features, np.ones(remaining_features.shape[0])))
        # Nx4 matrices
        features_coord_in_cam1 = (H_world2cam1 @ features_homo_coord.T).T
        # filter out points behind the camera
        valid_cam_1 = features_coord_in_cam1[:,2] > 0
        features_coord_in_cam1 = features_coord_in_cam1[valid_cam_1]
        remaining_features = remaining_features[valid_cam_1]
        index_of_valid_features = index_of_valid_features[valid_cam_1]

        # step: take the top (closest) max_count features
        if remaining_features.shape[0] > self.max_count:
            print('number of features in front of camera > threshold, the number is: ', remaining_features.shape[0], '. Only using the closest ', self.max_count, ' features.')
            diff = remaining_features - odom_position
            diff_values = np.linalg.norm(diff, axis = 1)
            sort_idx = np.argsort(diff_values)
            remaining_features = remaining_features[sort_idx[:self.max_count], :]
            features_coord_in_cam1 = features_coord_in_cam1[sort_idx[:self.max_count], :]
            index_of_valid_features = index_of_valid_features[sort_idx[:self.max_count]]

        # step: project feature to camera, check it is within camera's field of view
        # get pose from simulator, save the transformation between each camera and robot body frame for students
        extend_fov = 200 # in pixels
        cam_width = self.camera_matrix[0,2] * 2 + extend_fov
        cam_height = self.camera_matrix[1,2] * 2 + extend_fov
        K_cam1 = self.camera_matrix

        # projecting features onto images
        features_cam1 = (K_cam1 @ features_coord_in_cam1.T).T
        features_cam1[:,0] = features_cam1[:,0] / features_cam1[:,2]
        features_cam1[:,1] = features_cam1[:,1] / features_cam1[:,2]
        features_cam1[:,2] = features_cam1[:,2] / features_cam1[:,2]

        # filter out points outside FOV
        cam1_check_height = (features_cam1[:,1] < cam_height) & (features_cam1[:,1] > -extend_fov)
        cam1_check_width = (features_cam1[:,0] < cam_width) & (features_cam1[:,0] > -extend_fov)

        valid_features=  cam1_check_height & cam1_check_width
        # output valid features in previous step, arrange them to be Nx4 array, where:
        # (1) 1st-2nd columns are features in the left camera
        # (1) 3rd-4th columns are features in the right camera
        valid_features_2d = features_cam1[valid_features, :2]
        features_coord_in_cam1=  features_coord_in_cam1[valid_features,:]
        remaining_features = remaining_features[valid_features, :]
        index_of_valid_features = index_of_valid_features[valid_features]

        # step: the line connecting camera and feature does not intersect any obstacles
        visibility = np.full(remaining_features.shape[0], True) # self.check_visibility(remaining_features, odom_position)
        remaining_features = remaining_features[visibility, :]
        index_of_valid_features = index_of_valid_features[visibility]
        features_coord_in_cam1 = features_coord_in_cam1[visibility, :]
        valid_features_2d = valid_features_2d[visibility,:]

        if self.visualize:
            self.visualize_counter+=1
            if self.visualize_counter % self.visualize_interval == 0:
                visualize_3d = True
                dir = "/home/sam/meam620-final-project-demos/stereo/"
                if not os.path.isdir(dir):
                    print('change figure save directory to your own directory!!!')
                if visualize_3d:
                    valid_features_3d = remaining_features
                    camera_origin = np.linalg.pinv(H_world2cam1) @ np.array([0,0,0,1])
                    camera_z_axis = np.linalg.pinv(H_world2cam1) @ np.array([0,0,1,1])
                    fig = plt.figure('Features visualization')
                    ax = Axes3Ds(fig)
                    ax.scatter(camera_origin[0], camera_origin[1], camera_origin[2], c='black', s=150)
                    ax.plot3D([camera_origin[0],camera_z_axis[0]], [camera_origin[1],camera_z_axis[1]], [camera_origin[2],camera_z_axis[2]], c='black',linewidth=5)
                    ax.scatter(camera_z_axis[0], camera_z_axis[1], camera_z_axis[2], c='orange', s=150, marker='*')

                    # show the depth by marker size (this is slow)
                    visualize_depth = False
                    if visualize_depth:
                        for i in np.arange(valid_features_3d.shape[0]):
                            ax.scatter(valid_features_3d[i, 0], valid_features_3d[i, 1], valid_features_3d[i, 2], c='red', s=10*(valid_features_depth[i]-np.min(valid_features_depth)))
                    else:
                        ax.scatter(valid_features_3d[:, 0], valid_features_3d[:, 1], valid_features_3d[:, 2], c='red', s = 10)
                    ax.scatter( feature_positions[dist_within_thre, 0],  feature_positions[dist_within_thre, 1],  feature_positions[dist_within_thre, 2], c='black', s=0.1)

                    if self.plot_initialized==False:
                        self.plot_initialized = True
                    self.world.draw(ax)
                    if os.path.isdir(dir):
                        plt.savefig(fname=dir + "world_" + str(self.visualize_counter) + '.png')

                fig2 = plt.figure('Features in images')
                plt.scatter(valid_features_2d[:,0],valid_features_2d[:,1], c='blue',  s=3)
                plt.xlim(-100, cam_width+100)
                plt.ylim(-100, cam_height+100)
                plt.pause(0.001)
                if os.path.isdir(dir):
                    plt.savefig(fname = dir+"features_"+str(self.visualize_counter)+'.png')
                fig2.clf()

        valid_features_depth = features_coord_in_cam1[:,2]
        num_valid_features = valid_features_depth.shape[0]
        if num_valid_features == 0:
            print("no valid feature found!!")

        # those are what we need for running VIO (u v and depth of features in camera 1)
        # self.valid_features_uvd_cam1 = np.column_stack((valid_features_2d, valid_features_depth))
        valid_features_inverse_depth = 1.0 / valid_features_depth
        fx = K_cam1[0,0]
        fy = K_cam1[1,1]
        cx = K_cam1[0,2]
        cy = K_cam1[1,2]
        valid_features_normalized_u =(valid_features_2d[:,0] - cx) / fx
        valid_features_normalized_v =(valid_features_2d[:,1] - cy) / fy
        return np.column_stack((valid_features_normalized_u,valid_features_normalized_v)), valid_features_inverse_depth, remaining_features, index_of_valid_features#features_coord_in_cam1[valid_features,:3]


    def check_visibility(self, feature_positions_temp1, odom_position):
        blocks = self.all_blocks_coord
        visibility = np.full(feature_positions_temp1.shape[0], True)
        for idx, feature_position in enumerate(feature_positions_temp1):
            for block in blocks:
                intersection_found = CheckLineBox(block[0],block[1], odom_position, feature_position)
                if intersection_found:
                    visibility[idx] = 0
                    # no need to check again for this feature, it is not visible
                    break
        return visibility


# credit: this function is translated and modified based on http://www.3dkingdoms.com/weekly/weekly.php?a=3
def GetIntersection(fDst1, fDst2, P1, P2):
    if ( (fDst1 * fDst2) >= 0.0):
        return 0
    if ( fDst1 == fDst2):
        return 0
    global Hit
    Hit = P1 + (P2-P1) * ( -fDst1/(fDst2-fDst1) )
    return 1


# credit:this function is translated and modified based on http://www.3dkingdoms.com/weekly/weekly.php?a=3
def InBox(B1, B2, Axis):
    global Hit
    if ( Axis==1 and Hit[2] > B1[2] and Hit[2] < B2[2] and Hit[1] > B1[1] and Hit[1] < B2[1]):
        return 1
    if ( Axis==2 and Hit[2] > B1[2] and Hit[2] < B2[2] and Hit[0] > B1[0] and Hit[0] < B2[0]):
        return 1
    if ( Axis==3 and Hit[0] > B1[0] and Hit[0] < B2[0] and Hit[1] > B1[1] and Hit[1] < B2[1]):
        return 1
    return 0


# returns true if line (L1, L2) intersects with the box (B1, B2)
# credit: this function is translated and modified based on http://www.3dkingdoms.com/weekly/weekly.php?a=3
def CheckLineBox(B1, B2, L1, L2):
    if ((L2[0] < B1[0] and L1[0] < B1[0]) or  (L2[0] > B2[0] and L1[0] > B2[0])
        or (L2[1] < B1[1] and L1[1] < B1[1]) or (L2[1] > B2[1] and L1[1] > B2[1])
        or  (L2[2] < B1[2] and L1[2] < B1[2]) or (L2[2] > B2[2] and L1[2] > B2[2])):
        return False

    if ( (GetIntersection( L1[0]-B1[0], L2[0]-B1[0], L1, L2) and InBox(B1, B2, 1 ))
    or (GetIntersection( L1[1]-B1[1], L2[1]-B1[1], L1, L2) and InBox(B1, B2, 2 ))
    or (GetIntersection( L1[2]-B1[2], L2[2]-B1[2], L1, L2) and InBox(B1, B2, 3 ))
    or (GetIntersection( L1[0]-B2[0], L2[0]-B2[0], L1, L2) and InBox(B1, B2, 1 ))
    or (GetIntersection( L1[1]-B2[1], L2[1]-B2[1], L1, L2) and InBox(B1, B2, 2 ))
    or (GetIntersection( L1[2]-B2[2], L2[2]-B2[2], L1, L2) and InBox(B1, B2, 3 ))):
        return True

    return False

