# Littlegreen Humanoid Lite Assets

This package contains the asset files and tools used by Littlegreen Humanoid Lite. Legacy Berkeley-named robot layers are retained for upstream compatibility. It includes scripts for generating URDF files from Onshape and converting them to MJCF and USD formats to use in multiple simulators.


## Generate URDF from Onshape

To export the Onshape design as URDF file, we will be using the onshape-to-robot tool.

First, we need to install the necessary dependencies:

```bash
sudo apt install openscad meshlab
```

Optionally, activate the conda environment.

```bash
conda activate littlegreen-humanoid-lite
```

Then, clone and install this fork of onshape-to-robot:

```bash
git clone git@github.com:T-K-233/onshape-to-robot.git
cd ./onshape-to-robot/
pip install -e .
```

Finally, we can run the following command to generate the URDF file.

> **Note**
>
> During the generation process, onshape-to-robot will dump the intermediary files directly under the root of this repository. This is normal. After the conversion process is done, we have some scripts to automatically move them to the corresponding locations.

```bash
make urdf
```

The resulting URDF will be generated at `/data/urdf/` folder, and the STL meshes will be located under `/data/meshes/`.

## Convert URDF to MJCF

First, set the `MUJOCO_HOME` environment variable to point to the path of the mujoco binary.

An example would be:

```bash
export MUJOCO_HOME=/home/tk/Documents/mujoco-3.2.6/bin/
```

Then, run this command to convert URDF file to MJCF file.

```bash
make mjcf
```

We are not done yet. We will need to perform some manual post processing to the generated MJCF file, located under `/data/mjcf/`.

We need to add the following attributes to the generated mjcf file.

For the biped robot:

```xml
  ... <!-- generated content -->
  <worldbody>
    <body name="base" pos="0 0 0.65">
      <freejoint/>
      <site name="imu" size="0.01" pos="0 0 0" />
      ... <!-- generated content -->
    </body>
  </worldbody>

  <actuator>
    <motor name="leg_left_hip_roll"     gear="1" joint="leg_left_hip_roll_joint"/>
    <motor name="leg_left_hip_yaw"      gear="1" joint="leg_left_hip_yaw_joint"/>
    <motor name="leg_left_hip_pitch"    gear="1" joint="leg_left_hip_pitch_joint"/>
    <motor name="leg_left_knee_pitch"   gear="1" joint="leg_left_knee_pitch_joint"/>
    <motor name="leg_left_ankle_pitch"  gear="1" joint="leg_left_ankle_pitch_joint"/>
    <motor name="leg_left_ankle_roll"   gear="1" joint="leg_left_ankle_roll_joint"/>
    <motor name="leg_right_hip_roll"    gear="1" joint="leg_right_hip_roll_joint"/>
    <motor name="leg_right_hip_yaw"     gear="1" joint="leg_right_hip_yaw_joint"/>
    <motor name="leg_right_hip_pitch"   gear="1" joint="leg_right_hip_pitch_joint"/>
    <motor name="leg_right_knee_pitch"  gear="1" joint="leg_right_knee_pitch_joint"/>
    <motor name="leg_right_ankle_pitch" gear="1" joint="leg_right_ankle_pitch_joint"/>
    <motor name="leg_right_ankle_roll"  gear="1" joint="leg_right_ankle_roll_joint"/>
  </actuator>

  <sensor>
    <jointpos name="leg_left_hip_roll_pos"      joint="leg_left_hip_roll_joint"/>
    <jointpos name="leg_left_hip_yaw_pos"       joint="leg_left_hip_yaw_joint"/>
    <jointpos name="leg_left_hip_pitch_pos"     joint="leg_left_hip_pitch_joint"/>
    <jointpos name="leg_left_knee_pitch_pos"    joint="leg_left_knee_pitch_joint"/>
    <jointpos name="leg_left_ankle_pitch_pos"   joint="leg_left_ankle_pitch_joint"/>
    <jointpos name="leg_left_ankle_roll_pos"    joint="leg_left_ankle_roll_joint"/>
    <jointpos name="leg_right_hip_roll_pos"     joint="leg_right_hip_roll_joint"/>
    <jointpos name="leg_right_hip_yaw_pos"      joint="leg_right_hip_yaw_joint"/>
    <jointpos name="leg_right_hip_pitch_pos"    joint="leg_right_hip_pitch_joint"/>
    <jointpos name="leg_right_knee_pitch_pos"   joint="leg_right_knee_pitch_joint"/>
    <jointpos name="leg_right_ankle_pitch_pos"  joint="leg_right_ankle_pitch_joint"/>
    <jointpos name="leg_right_ankle_roll_pos"   joint="leg_right_ankle_roll_joint"/>

    <jointvel name="leg_left_hip_roll_vel"      joint="leg_left_hip_roll_joint"/>
    <jointvel name="leg_left_hip_yaw_vel"       joint="leg_left_hip_yaw_joint"/>
    <jointvel name="leg_left_hip_pitch_vel"     joint="leg_left_hip_pitch_joint"/>
    <jointvel name="leg_left_knee_pitch_vel"    joint="leg_left_knee_pitch_joint"/>
    <jointvel name="leg_left_ankle_pitch_vel"   joint="leg_left_ankle_pitch_joint"/>
    <jointvel name="leg_left_ankle_roll_vel"    joint="leg_left_ankle_roll_joint"/>
    <jointvel name="leg_right_hip_roll_vel"     joint="leg_right_hip_roll_joint"/>
    <jointvel name="leg_right_hip_yaw_vel"      joint="leg_right_hip_yaw_joint"/>
    <jointvel name="leg_right_hip_pitch_vel"    joint="leg_right_hip_pitch_joint"/>
    <jointvel name="leg_right_knee_pitch_vel"   joint="leg_right_knee_pitch_joint"/>
    <jointvel name="leg_right_ankle_pitch_vel"  joint="leg_right_ankle_pitch_joint"/>
    <jointvel name="leg_right_ankle_roll_vel"   joint="leg_right_ankle_roll_joint"/>

    <jointactuatorfrc name="leg_left_hip_roll_torque"     joint="leg_left_hip_roll_joint"/>
    <jointactuatorfrc name="leg_left_hip_yaw_torque"      joint="leg_left_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_left_hip_pitch_torque"    joint="leg_left_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_left_knee_pitch_torque"   joint="leg_left_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_pitch_torque"  joint="leg_left_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_roll_torque"   joint="leg_left_ankle_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_roll_torque"    joint="leg_right_hip_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_yaw_torque"     joint="leg_right_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_right_hip_pitch_torque"   joint="leg_right_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_right_knee_pitch_torque"  joint="leg_right_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_pitch_torque" joint="leg_right_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_roll_torque"  joint="leg_right_ankle_roll_joint"/>

    <framequat name="imu_quat" objtype="site" objname="imu" />
    <gyro name="imu_gyro" site="imu" />
    <accelerometer name="imu_acc" site="imu" />
    <framepos name="frame_pos" objtype="site" objname="imu" />
    <framelinvel name="frame_vel" objtype="site" objname="imu" />
  </sensor>
</mujoco>
```

For the humanoid robot:

```xml
  ... <!-- generated content -->
  <worldbody>
    <body name="base" pos="0 0 0.65">
      <freejoint/>
      <site name="imu" size="0.01" pos="0 0 0" />
      ... <!-- generated content -->
    </body>
  </worldbody>

  <actuator>
    <motor name="arm_left_shoulder_pitch"   gear="1" joint="arm_left_shoulder_pitch_joint"/>
    <motor name="arm_left_shoulder_roll"    gear="1" joint="arm_left_shoulder_roll_joint"/>
    <motor name="arm_left_shoulder_yaw"     gear="1" joint="arm_left_shoulder_yaw_joint"/>
    <motor name="arm_left_elbow_pitch"      gear="1" joint="arm_left_elbow_pitch_joint"/>
    <motor name="arm_left_elbow_roll"       gear="1" joint="arm_left_elbow_roll_joint"/>
    <motor name="arm_right_shoulder_pitch"  gear="1" joint="arm_right_shoulder_pitch_joint"/>
    <motor name="arm_right_shoulder_roll"   gear="1" joint="arm_right_shoulder_roll_joint"/>
    <motor name="arm_right_shoulder_yaw"    gear="1" joint="arm_right_shoulder_yaw_joint"/>
    <motor name="arm_right_elbow_pitch"     gear="1" joint="arm_right_elbow_pitch_joint"/>
    <motor name="arm_right_elbow_roll"      gear="1" joint="arm_right_elbow_roll_joint"/>
    <motor name="leg_left_hip_roll"         gear="1" joint="leg_left_hip_roll_joint"/>
    <motor name="leg_left_hip_yaw"          gear="1" joint="leg_left_hip_yaw_joint"/>
    <motor name="leg_left_hip_pitch"        gear="1" joint="leg_left_hip_pitch_joint"/>
    <motor name="leg_left_knee_pitch"       gear="1" joint="leg_left_knee_pitch_joint"/>
    <motor name="leg_left_ankle_pitch"      gear="1" joint="leg_left_ankle_pitch_joint"/>
    <motor name="leg_left_ankle_roll"       gear="1" joint="leg_left_ankle_roll_joint"/>
    <motor name="leg_right_hip_roll"        gear="1" joint="leg_right_hip_roll_joint"/>
    <motor name="leg_right_hip_yaw"         gear="1" joint="leg_right_hip_yaw_joint"/>
    <motor name="leg_right_hip_pitch"       gear="1" joint="leg_right_hip_pitch_joint"/>
    <motor name="leg_right_knee_pitch"      gear="1" joint="leg_right_knee_pitch_joint"/>
    <motor name="leg_right_ankle_pitch"     gear="1" joint="leg_right_ankle_pitch_joint"/>
    <motor name="leg_right_ankle_roll"      gear="1" joint="leg_right_ankle_roll_joint"/>
  </actuator>

  <sensor>
    <jointpos name="arm_left_shoulder_pitch_pos"  joint="arm_left_shoulder_pitch_joint"/>
    <jointpos name="arm_left_shoulder_roll_pos"   joint="arm_left_shoulder_roll_joint"/>
    <jointpos name="arm_left_shoulder_yaw_pos"    joint="arm_left_shoulder_yaw_joint"/>
    <jointpos name="arm_left_elbow_pitch_pos"     joint="arm_left_elbow_pitch_joint"/>
    <jointpos name="arm_left_elbow_roll_pos"      joint="arm_left_elbow_roll_joint"/>
    <jointpos name="arm_right_shoulder_pitch_pos" joint="arm_right_shoulder_pitch_joint"/>
    <jointpos name="arm_right_shoulder_roll_pos"  joint="arm_right_shoulder_roll_joint"/>
    <jointpos name="arm_right_shoulder_yaw_pos"   joint="arm_right_shoulder_yaw_joint"/>
    <jointpos name="arm_right_elbow_pitch_pos"    joint="arm_right_elbow_pitch_joint"/>
    <jointpos name="arm_right_elbow_roll_pos"     joint="arm_right_elbow_roll_joint"/>
    <jointpos name="leg_left_hip_roll_pos"        joint="leg_left_hip_roll_joint"/>
    <jointpos name="leg_left_hip_yaw_pos"         joint="leg_left_hip_yaw_joint"/>
    <jointpos name="leg_left_hip_pitch_pos"       joint="leg_left_hip_pitch_joint"/>
    <jointpos name="leg_left_knee_pitch_pos"      joint="leg_left_knee_pitch_joint"/>
    <jointpos name="leg_left_ankle_pitch_pos"     joint="leg_left_ankle_pitch_joint"/>
    <jointpos name="leg_left_ankle_roll_pos"      joint="leg_left_ankle_roll_joint"/>
    <jointpos name="leg_right_hip_roll_pos"       joint="leg_right_hip_roll_joint"/>
    <jointpos name="leg_right_hip_yaw_pos"        joint="leg_right_hip_yaw_joint"/>
    <jointpos name="leg_right_hip_pitch_pos"      joint="leg_right_hip_pitch_joint"/>
    <jointpos name="leg_right_knee_pitch_pos"     joint="leg_right_knee_pitch_joint"/>
    <jointpos name="leg_right_ankle_pitch_pos"    joint="leg_right_ankle_pitch_joint"/>
    <jointpos name="leg_right_ankle_roll_pos"     joint="leg_right_ankle_roll_joint"/>

    <jointvel name="arm_left_shoulder_pitch_vel"  joint="arm_left_shoulder_pitch_joint"/>
    <jointvel name="arm_left_shoulder_roll_vel"   joint="arm_left_shoulder_roll_joint"/>
    <jointvel name="arm_left_shoulder_yaw_vel"    joint="arm_left_shoulder_yaw_joint"/>
    <jointvel name="arm_left_elbow_pitch_vel"     joint="arm_left_elbow_pitch_joint"/>
    <jointvel name="arm_left_elbow_roll_vel"      joint="arm_left_elbow_roll_joint"/>
    <jointvel name="arm_right_shoulder_pitch_vel" joint="arm_right_shoulder_pitch_joint"/>
    <jointvel name="arm_right_shoulder_roll_vel"  joint="arm_right_shoulder_roll_joint"/>
    <jointvel name="arm_right_shoulder_yaw_vel"   joint="arm_right_shoulder_yaw_joint"/>
    <jointvel name="arm_right_elbow_pitch_vel"    joint="arm_right_elbow_pitch_joint"/>
    <jointvel name="arm_right_elbow_roll_vel"     joint="arm_right_elbow_roll_joint"/>
    <jointvel name="leg_left_hip_roll_vel"        joint="leg_left_hip_roll_joint"/>
    <jointvel name="leg_left_hip_yaw_vel"         joint="leg_left_hip_yaw_joint"/>
    <jointvel name="leg_left_hip_pitch_vel"       joint="leg_left_hip_pitch_joint"/>
    <jointvel name="leg_left_knee_pitch_vel"      joint="leg_left_knee_pitch_joint"/>
    <jointvel name="leg_left_ankle_pitch_vel"     joint="leg_left_ankle_pitch_joint"/>
    <jointvel name="leg_left_ankle_roll_vel"      joint="leg_left_ankle_roll_joint"/>
    <jointvel name="leg_right_hip_roll_vel"       joint="leg_right_hip_roll_joint"/>
    <jointvel name="leg_right_hip_yaw_vel"        joint="leg_right_hip_yaw_joint"/>
    <jointvel name="leg_right_hip_pitch_vel"      joint="leg_right_hip_pitch_joint"/>
    <jointvel name="leg_right_knee_pitch_vel"     joint="leg_right_knee_pitch_joint"/>
    <jointvel name="leg_right_ankle_pitch_vel"    joint="leg_right_ankle_pitch_joint"/>
    <jointvel name="leg_right_ankle_roll_vel"     joint="leg_right_ankle_roll_joint"/>

    <jointactuatorfrc name="arm_left_shoulder_pitch_torque"   joint="arm_left_shoulder_pitch_joint"/>
    <jointactuatorfrc name="arm_left_shoulder_roll_torque"    joint="arm_left_shoulder_roll_joint"/>
    <jointactuatorfrc name="arm_left_shoulder_yaw_torque"     joint="arm_left_shoulder_yaw_joint"/>
    <jointactuatorfrc name="arm_left_elbow_pitch_torque"      joint="arm_left_elbow_pitch_joint"/>
    <jointactuatorfrc name="arm_left_elbow_roll_torque"       joint="arm_left_elbow_roll_joint"/>
    <jointactuatorfrc name="arm_right_shoulder_pitch_torque"  joint="arm_right_shoulder_pitch_joint"/>
    <jointactuatorfrc name="arm_right_shoulder_roll_torque"   joint="arm_right_shoulder_roll_joint"/>
    <jointactuatorfrc name="arm_right_shoulder_yaw_torque"    joint="arm_right_shoulder_yaw_joint"/>
    <jointactuatorfrc name="arm_right_elbow_pitch_torque"     joint="arm_right_elbow_pitch_joint"/>
    <jointactuatorfrc name="arm_right_elbow_roll_torque"      joint="arm_right_elbow_roll_joint"/>
    <jointactuatorfrc name="leg_left_hip_roll_torque"         joint="leg_left_hip_roll_joint"/>
    <jointactuatorfrc name="leg_left_hip_yaw_torque"          joint="leg_left_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_left_hip_pitch_torque"        joint="leg_left_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_left_knee_pitch_torque"       joint="leg_left_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_pitch_torque"      joint="leg_left_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_left_ankle_roll_torque"       joint="leg_left_ankle_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_roll_torque"        joint="leg_right_hip_roll_joint"/>
    <jointactuatorfrc name="leg_right_hip_yaw_torque"         joint="leg_right_hip_yaw_joint"/>
    <jointactuatorfrc name="leg_right_hip_pitch_torque"       joint="leg_right_hip_pitch_joint"/>
    <jointactuatorfrc name="leg_right_knee_pitch_torque"      joint="leg_right_knee_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_pitch_torque"     joint="leg_right_ankle_pitch_joint"/>
    <jointactuatorfrc name="leg_right_ankle_roll_torque"      joint="leg_right_ankle_roll_joint"/>

    <framequat name="imu_quat" objtype="site" objname="imu" />
    <gyro name="imu_gyro" site="imu" />
    <accelerometer name="imu_acc" site="imu" />
    <framepos name="frame_pos" objtype="site" objname="imu" />
    <framelinvel name="frame_vel" objtype="site" objname="imu" />
  </sensor>
```


## Convert URDF to USD

Similar to the Mujoco flow, we need to set the `ISAACLAB_HOME` environment variable to the path of the IsaacLab repository.

An example would be:

```bash
export ISAACLAB_HOME=/home/tk/Documents/IsaacLab
```

> **Note**
>
> At the time of writing, the URDF -> USD conversion is not very stable in Isaac Lab >= 2.0. As a workaround, we are using Isaac Lab 1.4.0 to generate the USD file. A separate conda environment is required to run the conversion, and the `ISAACLAB_HOME` environment variable should point to the path of the Isaac Lab 1.4.0 repository.

Then, run the following command to convert the URDF to USD

```bash
make usd
```


## Axis Convention

We follow the ROS [REP-0103](https://www.ros.org/reps/rep-0103.html) standard for the frame and axis convention.

The forward direction of the robot is positive `X` axis, and the left direction is positive `Y` axis. The robot is standing still on a flat surface, with the `Z` axis pointing upwards.

The joint axis are defined to be pointing in the same direction as the robot's reference frame.


## Degree of Freedoms

| Joint ID | Name                           | CAN ID | Range      | Description                                                                                               |
| -------- | ------------------------------ | ------ | ---------- | --------------------------------------------------------------------------------------------------------- |
|  **Left Arm**                           | |        |            |                                                                                                           |
| 0        | arm_left_shoulder_pitch_joint  | 1      | [-90, 45]  | controls the flexion/extension (pitch) motion of the left upper arm. Positive is flexion                  |
| 1        | arm_left_shoulder_roll_joint   | 3      | [0, 90]    | controls the abduction/adduction (yaw) motion of the left upper arm. Positive is adduction                |
| 2        | arm_left_shoulder_yaw_joint    | 5      | [-45, 45]  | controls the rotation (roll) motion of the left upper arm. Positive is lateral rotation                   |
| 3        | arm_left_elbow_pitch_joint     | 7      | [-90, 0]   | controls the flexion/extension (pitch) motion of the left forearm. Positive is extension                  |
| 4        | arm_left_elbow_roll_joint      | 9      | [-45, 45]  | controls the rotation (roll) motion of the left forearm. Positive is lateral rotation                     |
|  **Right Arm**                          | |        |            |                                                                                                           |
| 5        | arm_right_shoulder_pitch_joint | 2      | [-45, 90]  | controls the flexion/extension (pitch) motion of the right upper arm. Positive is extension               |
| 6        | arm_right_shoulder_roll_joint  | 4      | [-90, 0]   | controls the abduction/adduction (yaw) motion of the right upper arm. Positive is abduction               |
| 7        | arm_right_shoulder_yaw_joint   | 6      | [-45, 45]  | controls the rotation (roll) motion of the right upper arm. Positive is medial rotation                   |
| 8        | arm_right_elbow_pitch_joint    | 8      | [-90, 0]   | controls the flexion/extension (pitch) motion of the right forearm. Positive is flexion                   |
| 9        | arm_right_elbow_roll_joint     | 10     | [-45, 45]  | controls the rotation (roll) motion of the right forearm. Positive is medial rotation                     |
|  **Left Leg**                             |        |            |                                                                                                           |
| 10       | leg_left_hip_roll_joint        | 1      | [-10, 90]  | controls the flexion/extension (pitch) motion of the left thigh. Positive is flexion                      |
| 11       | leg_left_hip_yaw_joint         | 3      | [-33.75, 56.25]  | controls the abduction/adduction (yaw) motion of the left thigh. Positive is adduction              |
| 12       | leg_left_hip_pitch_joint       | 5      | [-108.75, 56.25] | controls the rotation (roll) motion of the left thigh. Positive is lateral rotation                 |
| 13       | leg_left_knee_pitch_joint      | 7      | [0, 140]   | controls the flexion/extension (pitch) motion of the left shin. Positive is extension                     |
| 14       | leg_left_ankle_pitch_joint     | 11     | [-45, 45]  | controls the rotation (roll) motion of the left shin. Positive is lateral rotation                        |
| 15       | leg_left_ankle_roll_joint      | 13     | [-15, 15]  | controls the inversion / eversion (roll) motion of the left foot. Positive is eversion                    |
| **Right Leg**                             |        |            |                                                                                                           |
| 16       | leg_right_hip_roll_joint       | 2      | [-90, 10]  | controls the flexion/extension (pitch) motion of the right thigh. Positive is extension                   |
| 17       | leg_right_hip_yaw_joint        | 4      | [-56.25, 33.75]  | controls the abduction/adduction (yaw) motion of the right thigh. Positive is abduction             |
| 18       | leg_right_hip_pitch_joint      | 6      | [-56.25, 108.75] | controls the rotation (roll) motion of the right thigh. Positive is medial rotation                 |
| 19       | leg_right_knee_pitch_joint     | 8      | [0, 140]  | controls the flexion/extension (pitch) motion of the right shin. Positive is flexion                       |
| 20       | leg_right_ankle_pitch_joint    | 12     | [-45, 45]  | controls the rotation (roll) motion of the right shin. Positive is medial rotation                        |
| 21       | leg_right_ankle_roll_joint     | 14     | [-15, 15]  | controls the inversion / eversion (roll) motion of the right foot. Positive is inversion                  |


> **Info**
>
> The descriptions are based from [this reference](https://courses.lumenlearning.com/suny-ap1/chapter/types-of-body-movements/).

