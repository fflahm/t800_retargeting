# t800_retargeting

## install
```shell
pip install -r requirements.txt
```

## process original grab data
```shell
cd manotorch
python process_grab_data.py
```

## retarget from human hand and visualize without object

```shell
bash human_to_robot_no_object.sh
```

## retarget from human hand and visualize with object

```shell
bash human_to_robot.sh
```

## retarget from object

```shell
bash object_to_robot.sh
```

## structure of .pkl

- robot
  - data: (N, dof) robot hand joints angles retargeted
  - meta_data
- hand
  - joints_to_retarget: (N, 21, 3) mano joints positions aligned
  - joints_target: (N, 21, 3) mano joints positions original
  - wrist_loc: (N, 3) wrist positions
  - wrist_rot: (N, 4) wrist rotations in wxyz
- object
  - name
  - loc
  - rot
- table
  - loc
  - rot
