# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

from littlegreen_humanoid_lite_lowlevel.robot.bimanual import Bimanual


robot = Bimanual()

try:
    while True:
        robot.check_connection()
except KeyboardInterrupt:
    print("Stopping robot due to keyboard interrupt.")

robot.stop()
