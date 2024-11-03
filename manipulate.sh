name="ability"
python3 run_manipulation.py \
  --pickle-path data/qpos/${name}_hand_poses.pkl \
  --robot-glob-path data/grab/hand_glob_poses.pkl \
  --object-path data/grab/object_poses.pkl \
  # --output-video-path data/grab/demo.mp4 \
  # --headless