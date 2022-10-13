from heapq import heappush, heappop  # Recommended.
import numpy as np


from collections import defaultdict

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def distance(a, b):
    return  np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
      AB = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2 + (start[2] - end[2]) ** 2)
      AC = np.sqrt((start[0] - point[0]) ** 2 + (start[1] - point[1]) ** 2 + (start[2] - point[2]) ** 2)
      cotheta = (start[0]*end[0] + start[1]*end[1] + start[2]*end[2]) / ( np.linalg.norm(np.array(start)) * np.linalg.norm(np.array(end)) )
      AD = AC*cotheta
      CD = np.sqrt( AC**2 - AD**2 )
      return CD

def rdp(pointList, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(pointList) - 1):
        d = point_line_distance(pointList[i], pointList[0], pointList[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        resultList = rdp(pointList[:index+1], epsilon)[:-1] + rdp(pointList[index:], epsilon)
    else:
        resultList = [pointList[0], pointList[-1]]

    return resultList

def find_neighbor(cel, occ_map):
    neighbor = np.array([[1, -1, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, -1, 0],
                         [0, 1, 0],
                         [-1, -1, 0],
                         [-1, 0, 0],
                         [-1, 1, 0],
                         [1, - 1, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, -1, 1],
                         [0, 1, 1],
                         [- 1, - 1, 1],
                         [-1, 0, 1],
                         [- 1, 1, 1],
                         [0, 0, 1],
                         [1, -1, -1],
                         [1, 0, -1],
                         [1, 1, -1],
                         [0, -1, -1],
                         [0, 1, -1],
                         [-1, - 1, - 1],
                         [- 1, 0, - 1],
                         [- 1, 1, - 1],
                         [0, 0, - 1]])
    all_neig = neighbor + cel
    all_neig = all_neig[np.all(all_neig >= 0, axis=1), :]
    graph_size = occ_map.map
    size = graph_size.shape
    all_neig = all_neig[np.all(all_neig < size, axis=1), :]
    x_column = all_neig[:, 0]
    y_column = all_neig[:, 1]
    z_column = all_neig[:, 2]
    all_neig = all_neig[np.where(graph_size[x_column, y_column, z_column] == 0)]

    return all_neig

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    explored_set = set()
    frontier = [(0, start_index)]

    dist_dict = defaultdict(lambda: float("inf"))
    dist_heuristic_dict = defaultdict(lambda: float("inf"))
    parent_dic = {}
    dist_dict[start_index] = 0
    dist_heuristic_dict[start_index] = np.linalg.norm((np.array(goal_index)) - np.array(start_index))

    while frontier:
        current_index = heappop(frontier)[1]
        if current_index == goal_index:
            path = []
            needed_val = goal_index
            path.append(goal)

            while needed_val is not None:
                parent = parent_dic[needed_val]
                if parent == start_index:
                    path.insert(0, start)
                    break
                path.insert(0, occ_map.index_to_metric_center(parent))
                needed_val = parent
            path = np.array(path)
            return path, len(explored_set)

        explored_set.add(current_index)
        neighbors = find_neighbor(np.array(current_index), occ_map)

        for each_neighbor in neighbors:
            each_neighbor_tup = tuple(each_neighbor)
            diff = each_neighbor - current_index
            distance = np.linalg.norm(diff)
            dist_heuristic = dist_heuristic_dict[current_index]
            dist_heuristic += distance
            dist = dist_heuristic

            if astar == True:
                diff2 = np.linalg.norm(each_neighbor - goal_index)
                dist = dist_heuristic + diff2

            if dist_heuristic < dist_heuristic_dict[each_neighbor_tup]:
                dist_heuristic_dict[each_neighbor_tup] = dist_heuristic
                dist_dict[each_neighbor_tup] = dist
                parent_dic[each_neighbor_tup] = current_index
                heappush(frontier, (dist, each_neighbor_tup))




