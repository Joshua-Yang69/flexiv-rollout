# CLAUDE.md

## Overview

The project is used for real world robot policy deployment. We use the FlexivXense (Flexiv Robotic Arm Rizon 4s and Xense Gripper) as the deployed robot.

## Structure

The codebase should be broken down into three part. For the first part, is the state perceiver, in this session, the server can gain the state of the robot and the perceptions from cameras and sensors, and the streamline data then save to the shared buffer or use C/S structure to send to the model server. The second part is for robot policy model server, it is loaded persistently and receive the perception information from the shared buffer or client, and then make inference, after that the third part is receive the model output to send the control instruction to the robot server.

## Some Code reference

First write a stub method for model server as the basic class and then realize the detailed deployment of ACT and vt-muse in references/models.

We use realsense D415 as the visual camera and use Xense as the visuotactile sensor, and the code reference is under robot-api, we use the flexiv-rdk to read the robot state and give signals to robot arm, we use xensesdk to read gripper state and send control signal. You should caution that the state covers the robotic arm and xense, which is a 7+1 dimension information, their state should be read under the very same time. 

- The code reference of Xense Gripper is `references/robot-api/r3kit/devices/gripper/xense`
- The code reference of Flexiv Robot Arm is `references/robot-api/r3kit/devices/robot/flexiv`, we use NRT_CARTESIAN_MOTION_FORCE mode to control the robot
- The code reference of Xense sensor readin is `references/robot-api/r3kit/devices/camera/xense`
- The code reference of Realsense D415 readin is `references/robot-api/r3kit/devices/camera/realsense/general.py`

## Code guidelines

1. We only reserve the function we need for the project from the code reference. And we should wrap the code ourself instead of using `r3kit` because it is so complicated.
2. For each part, first setup basic class for general config and then make detailed realization
3. All code should be under `rollout` and `utils`. The config of each part should also be disentangled and use `.yml`.
4. You should pay attention to the effectiveness of the system, we cannot tolerate the latency.