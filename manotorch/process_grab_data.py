import numpy as np
import plotly.graph_objects as go
import trimesh as tm
# from manotorch.manolayer import ManoLayer
from manopth.manolayer import ManoLayer
from utils import generate_mano_outputs, plot_mesh, show_plotly
from scipy.spatial.transform import Rotation
import copy
import pickle
from pathlib import Path


object_name = "apple"
data_path = "../data/grab/apple_lift.npz"      # path of original grab data
# data_path = "../data/grab/apple_eat_1.npz"
beta_path = "../data/grab/s1_rhand_betas.npy"  # path of human hand beta
object_path = Path(__file__).absolute().parent.parent / "assets" / "objects" / "apple.ply" # path of object mesh
human_joints_save_path = "../data/tmp/human_hand_joints.pkl"
hand_poses_save_path = "../data/tmp/hand_glob_poses.pkl"
object_poses_save_path = "../data/tmp/object_poses.pkl"

verbose = True
save = False
# mano_layer = ManoLayer(use_pca=False, flat_hand_mean=True)
mano_layer = ManoLayer(mano_root="data/mano_models", use_pca=True, ncomps=24)
faces = mano_layer.th_faces.numpy()
rotvec_mano2robot = [-1.2092,1.2092,1.2092]
rotvec_robot2mano = [1.2092,-1.2092,-1.2092]

def make_transform(tran, rotvec):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = tran
    return T

def rotvecs_to_quats(rotvec_seq, pre_rotvec=[0.0,0.0,0.0]):
    rot_seq = Rotation.from_rotvec(rotvec_seq)
    pre_seq = Rotation.from_rotvec(pre_rotvec)
    return (rot_seq * pre_seq).as_quat(scalar_first=True)

def load_grab_data(data_path=data_path, beta_path=beta_path, start_num=200, end_num=300, interval=1):
    data_dict = np.load(data_path, allow_pickle=True)
    rhand = data_dict["rhand"].item()['params']
    human_rot_seq = rhand['global_orient'][start_num:end_num:interval] # [N, 3]
    human_tran_seq = rhand['transl'][start_num:end_num:interval] # [N, 3]
    # pose_seq = rhand['fullpose'][start_num:end_num:interval] # [N, 45]
    pose_seq = rhand['hand_pose'][start_num:end_num:interval]
    beta = np.load(beta_path) # [10]

    # table_data = data_dict["table"]
    # print(table_data)
    object_data = data_dict["object"].item()['params']
    object_tran_seq = object_data["transl"][start_num:end_num:interval] # [N, 3]
    object_rot_seq = object_data["global_orient"][start_num:end_num:interval] # [N, 3]
    return pose_seq, beta, human_tran_seq, human_rot_seq, object_tran_seq, object_rot_seq

def align_hand_for_retargeting(pose_seq, beta, pose_align=rotvec_mano2robot, verbose=verbose):
    align_seq = np.tile(np.array(pose_align),[len(pose_seq),1]) # [N, 3]
    aligned_pose_seq = np.hstack([align_seq, pose_seq]) # [N, 48], align for retargeting
    verts_seq, joints_seq = generate_mano_outputs(mano_layer, aligned_pose_seq, beta)
    wrist_seq = joints_seq[:,:1,:] # [N, 1, 3]
    verts_seq = verts_seq - wrist_seq # [N, 778, 3]
    joints_seq = joints_seq - wrist_seq # [N, 21, 3]
    if verbose:
        data_start = [plot_mesh(tm.Trimesh(verts_seq[0],faces),name="aligned_hand")]
        frames = []
        for verts in verts_seq:
            hand_fig = plot_mesh(tm.Trimesh(verts,faces),name="aligned_hand")
            frames.append(go.Frame(data=[hand_fig]))
        show_plotly(data_start,frames)
    return joints_seq

def process_human_seq(pose_seq, beta, tran_seq, rot_seq):
    full_pose_seq = np.hstack([rot_seq, pose_seq])
    verts_seq, joints_seq = generate_mano_outputs(mano_layer, full_pose_seq, beta)
    verts_seq = verts_seq + tran_seq[:, np.newaxis, :] # [N, 778, 3]
    joints_seq = joints_seq + tran_seq[:, np.newaxis, :] # [N, 21, 3]
    wrist_seq = joints_seq[:,0,:] # [N, 3]
    return verts_seq, wrist_seq

def visualize_seq(verts_seq, object_path, object_tran_seq, object_rot_seq):
    object_mesh_origin = tm.load(object_path)
    object_start = copy.deepcopy(object_mesh_origin)
    object_start.apply_transform(make_transform(object_tran_seq[0],object_rot_seq[0]))
    data_start = [plot_mesh(tm.Trimesh(verts_seq[0],faces),name="hand"),
                plot_mesh(object_start, color="lightblue", name="apple")]
    frames = []
    for verts, obj_tran, obj_rot in zip(verts_seq,object_tran_seq,object_rot_seq):
        hand_fig = plot_mesh(tm.Trimesh(verts,faces),name="hand")
        object_mesh = copy.deepcopy(object_mesh_origin)
        object_mesh.apply_transform(make_transform(obj_tran, obj_rot))
        object_fig = plot_mesh(object_mesh, color="lightblue", name="apple")
        frame = go.Frame(data=[hand_fig, object_fig])
        frames.append(frame)
    # show_plotly(data_start,frames,[[-1.0, 0.2],[-0.6, 0.6],[0.5, 1.7]])
    show_plotly(data_start,frames,[[-0.15, 0.05],[-0.56, -0.36],[0.8, 1.0]])

if __name__ == "__main__":
    # process
    pose_seq, beta, human_tran_seq, human_rot_seq, object_tran_seq, object_rot_seq = load_grab_data()
    joints_seq = align_hand_for_retargeting(pose_seq, beta) # [N, 21, 3]
    verts_seq, wrist_seq = process_human_seq(pose_seq, beta, human_tran_seq, human_rot_seq)
    if verbose:
        visualize_seq(verts_seq, object_path, object_tran_seq, object_rot_seq)
    
    # save
    if save:
        hand_quat_seq = rotvecs_to_quats(human_rot_seq, rotvec_robot2mano) # [N, 4]
        object_quat_seq = rotvecs_to_quats(object_rot_seq)
        with open(human_joints_save_path, "wb") as file:
            pickle.dump(joints_seq, file)
        with open(hand_poses_save_path, "wb") as file:
            pickle.dump({"tran_seq":wrist_seq,"rot_seq":hand_quat_seq}, file)
        with open(object_poses_save_path, "wb") as file:
            pickle.dump({"name":object_name,"tran_seq":object_tran_seq,"rot_seq":object_quat_seq}, file)