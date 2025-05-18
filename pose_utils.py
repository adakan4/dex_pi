import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')  # Set this BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm, colors

def poses7d_to_mats(poses):
    poses_mat = np.zeros((len(poses), 4, 4))
    
    for i, pose in enumerate(poses):
        poses_mat[i] = pose7d_to_mat(pose)
    
    return poses_mat

def mats_to_poses7d(mats):
    poses = np.zeros((len(mats), 7))
    
    for i, mat in enumerate(mats):
        poses[i] = mat_to_pose7d(mat)
    
    return poses

def abs_to_rel_poses(poses):
    rel_poses = np.zeros((len(poses), 4, 4))
    rel_poses[0] = np.eye(4)
    for i in range(1, len(poses)):
        rel_poses[i] = np.linalg.inv(poses[i-1]) @ poses[i]
    
    return rel_poses

def rel_to_abs_poses(rel_poses, first_pose=None):
    abs_poses = np.zeros((len(rel_poses), 4, 4))
    if first_pose is not None:
        abs_poses[0] = first_pose
    else:
        abs_poses[0] = np.eye(4)
    for i in range(1, len(abs_poses)):
        abs_poses[i] = abs_poses[i-1] @ rel_poses[i-1]
    
    return abs_poses
    
def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose9d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d9 = np.concatenate([pos, d6], axis=-1)
    return d9

def pose9d_to_mat(d9):
    pos = d9[...,:3]
    d6 = d9[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d9.shape[:-1]+(4,4), dtype=d9.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out

def pose9d_to_pose7d(d9):
    return mat_to_pose7d(pose9d_to_mat(d9))

def pose9d_to_pose6d(d9):
    return mat_to_pose6d(pose9d_to_mat(d9))

def pose7d_to_pose9d(d7):
    return mat_to_pose9d(pose7d_to_mat(d7))

def pose6d_to_pose9d(d6):
    return mat_to_pose9d(pose6d_to_mat(d6))

def pose7d_to_pose6d(d7):
    rotation = R.from_quat(d7[3:]).as_euler('xyz')
    return np.concatenate([d7[:3], rotation])

def poses7d_to_poses6d(poses7d):
    poses6d = np.zeros((len(poses7d), 6))
    for i, pose7d in enumerate(poses7d):
        poses6d[i] = pose7d_to_pose6d(pose7d)
    return poses6d

def pose6d_to_pose7d(d6):
    if d6 is None: return None
    rotation = R.from_euler('xyz', d6[3:]).as_quat()
    return np.concatenate([d6[:3], rotation])

def poses6d_to_poses7d(poses6d):
    poses7d = np.zeros((len(poses6d), 7))
    for i, pose6d in enumerate(poses6d):
        poses7d[i] = pose6d_to_pose7d(pose6d)
    return poses7d

def pose6d_to_mat(d6):
    pos = d6[:3]
    rot = R.from_euler('xyz', d6[3:]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat

def pose7d_to_mat(d7):
    pos = d7[:3]
    rot = R.from_quat(d7[3:]).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = pos
    return mat

def mat_to_pose6d(mat):
    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3]).as_euler('xyz')
    return np.concatenate([pos, rot])

def mat_to_pose7d(mat):
    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3]).as_quat()
    return np.concatenate([pos, rot])


def mat_to_pose_func(pose_repr="euler"):
    if pose_repr == "euler":
        return mat_to_pose6d
    elif pose_repr == "rot6d":
        return mat_to_pose9d
    elif pose_repr == "quat":
        return mat_to_pose7d
    else:
        raise ValueError(f"Unknown pose representation: {pose_repr}")

def pose_to_mat_func(pose_repr="euler"):
    if pose_repr == "euler":
        return pose6d_to_mat
    elif pose_repr == "rot6d":
        return pose9d_to_mat
    elif pose_repr == "quat":
        return pose7d_to_mat
    else:
        raise ValueError(f"Unknown pose representation: {pose_repr}")

def pose_to_pose6d(rot_repr="euler"):
    if rot_repr == "quat":
        return pose7d_to_pose6d
    elif rot_repr == "euler":
        def identity(pose):
            return pose
        return identity
    elif rot_repr == "rot6d":
        return pose9d_to_pose6d

def pose6d_to_pose(rot_repr="euler"):
    if rot_repr == "quat":
        return pose6d_to_pose7d
    elif rot_repr == "euler":
        def identity(pose):
            return pose
        return identity
    elif rot_repr == "rot6d":
        return pose6d_to_pose9d

    
def average_quaternions(quaternions, weights):
    weights = weights / np.sum(weights)
    A = np.zeros((4, 4))
    for q, w in zip(quaternions, weights):
        A += w * np.outer(q, q)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    mean_quaternion = eigenvectors[:, np.argmax(eigenvalues)]

    if mean_quaternion[0] < 0:
        mean_quaternion = -mean_quaternion

    mean_quaternion = mean_quaternion / np.linalg.norm(mean_quaternion)
    
    return mean_quaternion

def pose_distance(pose1, pose2):
    pos1, euler1 = np.array(pose1[:3]), np.radians(pose1[3:])
    pos2, euler2 = np.array(pose2[:3]), np.radians(pose2[3:])

    # Compute Euclidean distance for position
    d_pos = np.linalg.norm(pos1 - pos2)

    # Convert Euler angles to rotation matrices
    R1 = R.from_euler('xyz', euler1).as_matrix()
    R2 = R.from_euler('xyz', euler2).as_matrix()

    # Compute rotation difference matrix
    R_diff = R1 @ R2.T

    # Compute orientation distance (rotation angle)
    trace_value = np.trace(R_diff)
    trace_value = np.clip((trace_value - 1) / 2, -1.0, 1.0)  # Clip for numerical stability
    d_ori = np.arccos(trace_value)

    # Compute weighted total distance
    d_total = d_pos + d_ori

    return d_total

def to_intergripper_pose(right_poses, left_poses):
    inter_gripper_poses = []
    
    left_poses_copy = left_poses.copy()
    
    left_poses_copy[:, 1] += 0.46 # shift over to make it in the right frame
    
    # left_poses_copy[:, 1 ] += 0.35 # shift over less to make it seem closer
    
    for i in range(len(right_poses)):
        right_pose = right_poses[i]
        left_pose = left_poses_copy[i]
        
        right_pose_mat = pose6d_to_mat(right_pose)
        left_pose_mat = pose6d_to_mat(left_pose)
        
        inter_gripper_pose = np.linalg.inv(right_pose_mat) @ left_pose_mat
        
        inter_gripper_poses.append(mat_to_pose6d(inter_gripper_pose))
    
    return inter_gripper_poses


### PLOTTTING ####

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_one_path3d(tracked_pose):
    if len(tracked_pose) < 2:
        raise ValueError("tracked_pose must contain at least two points to create a path.")
        
    # Create segments
    points = tracked_pose
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

    # Normalize time steps to range [0, 1] for colormap
    timesteps = np.linspace(0, 1, len(points) - 1)
    
    # Get a colormap (e.g., "viridis")
    cmap = cm.get_cmap('viridis')
    colors_array = cmap(timesteps)

    # Create a Line3DCollection
    lc = Line3DCollection(segments, colors=colors_array, linewidth=2)
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Adjust the axes to equal scale
    
    # Set axis limits
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_zlim(points[:, 2].min(), points[:, 2].max())
    
    set_axes_equal(ax)
    
    norm = colors.Normalize(vmin=0, vmax=len(points) - 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Timestep')
    
    return fig

def plot_one_path3d(tracked_pose):
    fig = make_one_path3d(tracked_pose)
    plt.show()
    
def save_one_path3d(tracked_pose, save_path):
    fig = make_one_path3d(tracked_pose)
    plt.savefig(save_path)
    plt.close()

def make_paths3d(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, pose in enumerate(poses):
        ax.plot(pose[:, 0], pose[:, 1], pose[:, 2], label=f"Path {i}")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    return fig

def plot_paths3d(poses):
    fig = make_paths3d(poses)
    plt.show()

def save_paths3d(poses, save_path):
    fig = make_paths3d(poses)
    plt.savefig(save_path)
    plt.close()