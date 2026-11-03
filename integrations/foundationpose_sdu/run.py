"""Run one of the preserved SDU scripts against the pinned official source."""
import argparse
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
    parser.add_argument('script', choices=('grasp_config_ui.py', 'run_video_demo.py', 'run_single_pose_robot_demo.py'))
    args, remaining = parser.parse_known_args()
    official = path('foundationpose')
    sys.path[:0] = [str(path('native') / 'foundationpose'), str(HERE / 'files'), str(official)]
    install(official, ROOT / '.cache/foundationpose_sdu')
    sys.argv = [str(HERE / 'files' / args.script), *remaining]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
