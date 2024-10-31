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