name=svh
object_position_path=data/object_positions.pkl
object_pose_path=data/object_poses.pkl
robot_pose_path=data/${name}_hand_poses.pkl

python generate_object_poses.py \
  --position-path ${object_position_path} \
  --pose-path ${object_pose_path} \
  --seq-len 100
python retarget_from_joints.py \
  --robot-name $name \
  --input-path ${object_position_path} \
  --retargeting-type vector \
  --hand-type right \
  --output-path ${robot_pose_path} \
  --config-tag from_object_
python render_robot_hand.py \
  --pickle-path ${robot_pose_path} \
  --object-path ${object_pose_path}
  # --output-video-path data/tmp.mp4\
  # --headless