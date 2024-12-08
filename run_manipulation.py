from pathlib import Path
from typing import Optional, List, Union, Dict
import math
import cv2
import numpy as np
import sapien
import tqdm
import tyro
import pickle
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation

# Convert webp
# ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp


np.set_printoptions(precision=3)
pi = np.pi

mesh_dir = "grab/tools/object_meshes/contact_meshes"
fric = 0.5
resti = 0.0
robot_dens = 1000.0
object_dens = 400.0
stiff = 1000.0
damp = 50.0
scaling = 1.0
timestep = 1 / 120.0
interval = 1
if_create_object = False
if_pure_visual = False
if_show_markers = True
if_show_dists = True
if_align_root = True

def get_link_positions_from_names(robot, link_names):
    positions = []
    for name in link_names:
        positions.append(robot.find_link_by_name(name).get_entity_pose().get_p())
    return positions

def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    robot_glob_data: Dict,
    obj_data: Dict,
    table_data: Dict,
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):  
    # Generate rendering config
    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    scene.set_timestep(timestep)
    sapien.physx.set_default_material(fric, fric, resti)

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    # physx_mat: sapien.physx.PhysxMaterial = sapien.physx.PhysxMaterial(
    #     static_friction=0.5,
    #     dynamic_friction=0.5,
    #     restitution=0.0,
    # )

    table_mesh_path = f"{mesh_dir}/table.ply"
    table_tran = table_data["loc"]
    table_rot = table_data["rot"]
    table_builder = scene.create_actor_builder()
    table_builder.add_convex_collision_from_file(filename=table_mesh_path)
    table_builder.add_visual_from_file(filename=table_mesh_path, material=render_mat)
    table = table_builder.build_kinematic("table")
    table.set_pose(sapien.Pose(table_tran, table_rot))
    # scene.add_ground(table_height,  render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)
    
    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    # cam.set_local_pose(sapien.Pose([0.2, -0.45, 0.85], [0.0,0.0,0.0,-1.0]))
    cam.set_local_pose(sapien.Pose([0.4, -0.5, 0.85], [0.0,0.0,0.0,-1.0]))

    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None
    record_video = output_video_path is not None

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    loader.fix_root_link = True
    loader.scale = scaling
    loader.set_density(robot_dens)
    
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)
    
    for link in robot.links:
        link.disable_gravity = True

    # for link in robot.links:
    #     frics = []
    #     for ent in link.collision_shapes:
    #         frics.append(ent.density)
    #     print(link.name, frics)

    
    # Create object
    if if_create_object:
        obj_name = obj_data["name"]
        obj_mesh_path = f"{mesh_dir}/{obj_name}.ply"
        obj_tran_seq = obj_data["loc"] # [N, 3]
        obj_rot_seq = obj_data["rot"] # [N, 4]

        if (len(data) != len(obj_tran_seq)):
            raise ValueError("Hand and object length not match!")

        obj_builder = scene.create_actor_builder()
        obj_builder.add_convex_collision_from_file(filename=obj_mesh_path, density=object_dens)
        obj_builder.add_visual_from_file(filename=obj_mesh_path)
    
        object = obj_builder.build(name="object")
        
    # physx_obj:sapien.pysapien.physx.PhysxRigidDynamicComponent = object.components[1]
    # print(physx_obj.collision_shapes[0].physical_material.restitution)

    # Video recorder
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cam.get_width(), cam.get_height())
        )

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)
    

    link_names = config.target_task_link_names
    human_indices = config.target_link_human_indices[1]

    glob_tran_seq = robot_glob_data["wrist_loc"] # [N, 3]
    glob_rot_seq = robot_glob_data["wrist_rot"] # [N, 4]
    qpos_seq = np.array(data)[:,retargeting_to_sapien]
    target_positions_seq = robot_glob_data["joints_target"] # [N, 21, 3]
    target_positions_seq = target_positions_seq[:,human_indices,:] # [N, 5, 3]
    
    markers = []
    if if_show_markers:
        for link_name in link_names:
            marker_builder = scene.create_actor_builder()
            marker_builder.add_sphere_visual(radius=0.005)
            markers.append(marker_builder.build_kinematic(name=f"target_{link_name}"))

        marker_builder = scene.create_actor_builder()
        marker_builder.add_sphere_visual(radius=0.005)
        # bottom = marker_builder.build_kinematic(name="bottom")
        # bottom.set_pose(sapien.Pose([ 0.20784388,-0.5304037,0.7901369 ]))
        # bottom.set_pose(sapien.Pose([ 0.02140963,-0.51311964,0.7516789 ]))
    
    active_joints = robot.get_active_joints()
    
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=stiff, damping=damp)# , mode="force")

    while True:
        link_positions_seq = []
        if if_align_root:
            robot.set_root_pose(sapien.Pose(table_tran))
        else:
            robot.set_root_pose(sapien.Pose(glob_tran_seq[0],glob_rot_seq[0]))
        robot.set_qpos(qpos_seq[0])
        if if_create_object:
            object.set_pose(sapien.Pose(obj_tran_seq[0],obj_rot_seq[0]))
            # object.set_pose(sapien.Pose(obj_tran_seq[0],[0.70738827, 0.0, 0.0, -0.70682518]))
        if if_show_markers:
            for j in range(len(markers)):
                markers[j].set_pose(sapien.Pose(target_positions_seq[0][j]))
        if not headless:
            pause = True
            while not viewer.closed and pause:
                if viewer.window.key_down('c'):
                    pause = False
                scene.update_render()
                viewer.render()
        for i in tqdm.trange(len(data)):
            if not if_align_root:
                robot.set_root_pose(sapien.Pose(glob_tran_seq[i],glob_rot_seq[i]))
            if if_pure_visual:
                robot.set_qpos(qpos_seq[i])
                if if_create_object:
                    object.set_pose(sapien.Pose(obj_tran_seq[i],obj_rot_seq[i]))
            qf = robot.compute_passive_force()
            robot.set_qf(qf)
            for joint_idx, joint in enumerate(active_joints):
                joint.set_drive_target(qpos_seq[i][joint_idx])

            if if_show_markers:
                for j in range(len(markers)):
                    markers[j].set_pose(sapien.Pose(target_positions_seq[i][j]))
            
            for _ in range(interval):
                if not if_pure_visual:
                    scene.step()
                scene.update_render()
                if not headless:
                    viewer.render()
                if record_video:
                    cam.take_picture()
                    rgb = cam.get_picture("Color")[..., :3]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
            if if_show_dists:
                link_positions_seq.append(get_link_positions_from_names(robot,link_names))

        if if_show_dists:
            link_positions_seq = np.array(link_positions_seq) # [N, 6, 3]
            dists = np.linalg.norm(target_positions_seq-link_positions_seq,axis=2)
            dists = np.mean(dists, axis=0)
            print(f"distances: {dists}")
        
        if record_video:
            break

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str = "data/apple/svh_hand_poses.pkl",
    robot_glob_path:str = "data/apple/hand_glob_poses.pkl",
    object_path: str = "data/apple/object_poses.pkl",
    output_video_path: Optional[str] = None,
    headless: bool = False
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
    """
    robot_dir = "assets/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, robot_data = pickle_data["robot"]["meta_data"], pickle_data["robot"]["data"]
    robot_glob_data = pickle_data["hand"]
    object_data = pickle_data["object"]
    table_data = pickle_data["table"]
    render_by_sapien(meta_data, robot_data, robot_glob_data, object_data, table_data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
