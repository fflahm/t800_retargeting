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

# def process_rotations(origin_quat):
#     origin_rot = Rotation.from_rotvec(origin_quat)
#     trans_rot = Rotation.from_quat([0.5,-0.5,-0.5,0.5])
#     target_quat = (origin_rot*trans_rot).as_quat() # [x, y, z, w]
#     return target_quat[[3,0,1,2]]

np.set_printoptions(precision=3)
pi = np.pi

def get_link_positions_from_names(robot, link_names):
    positions = []
    for name in link_names:
        positions.append(robot.find_link_by_name(name).get_entity_pose().get_p())
    return positions

def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    robot_glob_path: str,
    obj_path: str,
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

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.2, -0.45, 0.85], [0.0,0.0,0.0,-1.0]))

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

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)
    
    # Create object
    obj_dir = "assets/objects"
    with open(obj_path,"rb") as file:
        obj_data = pickle.load(file)
        obj_name = obj_data["name"]
        obj_mesh_path = f"{obj_dir}/{obj_name}.ply"
        obj_tran_seq = obj_data["tran_seq"] # [N, 3]
        obj_rot_seq = obj_data["rot_seq"] # [N, 4]
    if (len(data) != len(obj_tran_seq)):
        raise ValueError("Hand and object length not match!")
    obj_builder = scene.create_actor_builder()
    obj_builder.add_convex_collision_from_file(filename=obj_mesh_path)
    obj_builder.add_visual_from_file(filename=obj_mesh_path)
    object = obj_builder.build(name="object")

    # for joint in robot.get_active_joints():
    #     print(joint.name)

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
    
    with open(robot_glob_path, "rb") as file:
        glob_data = pickle.load(file)
        glob_tran_seq = glob_data["tran_seq"] # [N, 3]
        glob_rot_seq = glob_data["rot_seq"] # [N, 4]
        target_positions_seq = glob_data["joints_seq"] # [N, 21, 3]
        target_positions_seq = target_positions_seq[:,[0,4,8,12,16,20],:] # [N, 6, 3]
    
    link_names = ["base_link","thumb_tip","index_tip","middle_tip","ring_tip","pinky_tip"]
    markers = []
    for i in range(len(link_names)):
        marker_builder = scene.create_actor_builder()
        marker_builder.add_sphere_visual(radius=0.005)
        markers.append(marker_builder.build(name=str(i)))
    def invert_quat(quat):
        return [quat[0], -quat[1], -quat[2], -quat[3]]
    
    link_positions_seq = []
    for i in tqdm.trange(len(data)):

        robot.set_pose(sapien.Pose(glob_tran_seq[i],glob_rot_seq[i]))
        robot.set_qpos(np.array(data[i])[retargeting_to_sapien])
        object.set_pose(sapien.Pose(obj_tran_seq[i],invert_quat(obj_rot_seq[i])))
        for j in range(len(markers)):
            markers[j].set_pose(sapien.Pose(target_positions_seq[i][j]))
        # link_positions_seq.append(get_link_positions_from_names(robot,link_names))
        if not headless:
            for _ in range(5):
                scene.update_render()
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])
    # link_positions_seq = np.array(link_positions_seq) # [N, 6, 3]
    # dists = np.linalg.norm(target_positions_seq-link_positions_seq,axis=2)
    # dists = np.mean(dists, axis=0)
    # print(f"distances: {dists}")


    if not headless:
        while not viewer.closed:
            viewer.render()

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    robot_glob_path: str,
    object_path: str,
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
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]


    render_by_sapien(meta_data, data, robot_glob_path, object_path, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
