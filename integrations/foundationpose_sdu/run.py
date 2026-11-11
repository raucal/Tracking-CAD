"""Run one of the preserved SDU scripts against the pinned official source."""
import argparse
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
from settings import path
from runtime import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Load the script without executing its main block or connecting hardware')
    parser.add_argument('--teleop-root', type=Path, default=os.environ.get('SDU_TELEOP_ROOT'), help='Directory containing the teleop_setup package used by the robot demo')
    parser.add_argument('script', choices=('grasp_config_ui.py', 'run_video_demo.py', 'run_single_pose_robot_demo.py'))
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remaining = args.script_args
    if args.teleop_root:
        sys.path.insert(0, str(Path(args.teleop_root).expanduser().resolve()))
    official = path('foundationpose')
    sys.path[:0] = [str(path('native') / 'foundationpose'), str(HERE / 'files'), str(official)]
    install(official, ROOT / '.cache/foundationpose_sdu')
    from foundationpose_sdu import configure_weights
    configure_weights()
    sys.argv = [str(HERE / 'files' / args.script), *remaining]
    runpy.run_path(sys.argv[0], run_name='__sdu_startup_check__' if args.check else '__main__')
    if args.check:
        print('SDU startup imports OK: ' + args.script, flush=True)


if __name__ == '__main__':
    main()
