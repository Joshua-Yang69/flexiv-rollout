"""
Usage:
    python homing.py -id <1 or 2>
"""
import argparse

_SN_TOOL_MAP = {
    "Rizon4s-063652": "tool2",   # master
    "Rizon4s-063586": "xense",   # slave
}


def home_robot(robot_sn: str, tool_name: str) -> None:
    """Home a single robot to its joint home position via the Rizon SDK."""
    from r3kit.devices.robot.flexiv.rizon import Rizon

    print(f"Homing {robot_sn} (tool={tool_name}) ...")
    robot = Rizon(id=robot_sn, gripper=False, name="Rizon4s", tool_name=tool_name)
    robot.motion_mode("joint")
    robot.homing()
    robot.motion_mode("primitive")
    print(f"Homing done: {robot_sn}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose which robot to home",
    )
    parser.add_argument(
        "-id",
        "--id",
        dest="robot_id",
        type=int,
        choices=[1, 2],
        required=True,
        help="1 refers to master robot, 2 refers to slave robot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.robot_id == 1:
        robot_sn = "Rizon4s-063652"
    elif args.robot_id == 2:
        robot_sn = "Rizon4s-063586"
    else:
        raise ValueError("Invalid robot ID")

    tool_name = _SN_TOOL_MAP[robot_sn]
    print(f"Homing robot {args.robot_id}: {robot_sn}")
    home_robot(robot_sn, tool_name)
    print("Homing command finished")


if __name__ == "__main__":
    main()

