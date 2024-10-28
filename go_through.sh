name="svh"
python generate_object_poses.py
python retarget_from_joints.py \
  --robot-name $name \
  --input-path data/object_positions.pkl \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/${name}_hand_poses.pkl
python3 render_robot_hand.py \
  --pickle-path data/${name}_hand_poses.pkl \
  --output-video-path data/tmp.mp4\
  --headless