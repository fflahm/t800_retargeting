name="shadow"
dir="mug"
python3 run_manipulation.py \
  --pickle-path data/grab/s1/apple_lift.pkl \
  --robot-glob-path data/${dir}/hand_glob_poses.pkl \
  --object-path data/${dir}/object_poses.pkl \
  # --output-video-path data/demo.mp4 \
  # --headless