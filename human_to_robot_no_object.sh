name="svh"
python retarget_from_joints.py \
  --robot-name $name \
  --input-path data/grab/human_hand_joints.pkl \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/qpos/${name}_hand_poses.pkl \
  --post-freq 1 \
  --config-tag from_human_ 
  
python3 render_robot_hand_no_object.py \
  --pickle-path data/qpos/${name}_hand_poses.pkl 