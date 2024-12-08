name="shadow"
python retarget_from_joints.py \
  --robot-name $name \
  --input-path data/apple/human_hand_joints.pkl \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/apple/${name}_hand_poses.pkl \
  # --config-tag from_human_ 
  
python3 render_robot_hand_no_object.py \
  --pickle-path data/apple/${name}_hand_poses.pkl 