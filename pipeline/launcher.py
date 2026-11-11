"""Portable launcher. --settings selects the runtime before importing application modules."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--settings', type=Path, help='JSON paths/runtime configuration')
    parser.add_argument('command', choices=('ui','suite','evaluate','detect','scores','sam','prepare','inspect','audit','install','install-foundationpose','settings','test'))
    if sys.argv[1:] in ([], ['--help'], ['-h']): parser.print_help(); return 0
    args, extra = parser.parse_known_args()
    if args.settings: os.environ['CAD_TRACKING_SETTINGS'] = str(args.settings.expanduser().resolve())
    import settings as cfg
    if args.command == 'settings':
        import json
        print(json.dumps({'settings_file':str(cfg.SETTINGS_FILE),'paths':{key:str(cfg.path(key)) for key in cfg.SETTINGS['paths']}},indent=2)); return 0
    python = Path(os.environ.get('CAD_TRACKING_PYTHON', cfg.path('runtime_python')))
    if not python.is_file(): parser.error(f'Runtime Python missing: {python}; edit config/settings.json')
    env = os.environ.copy()
    base = subprocess.check_output([str(python), '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()
    env['LD_LIBRARY_PATH'] = str(Path(base)/'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['PYTHONPATH'] = str(cfg.PIPELINE) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENCV_IO_ENABLE_OPENEXR='1')
    threads = str(cfg.SETTINGS['runtime']['threads'])
    for key,value in {'OMP_NUM_THREADS':threads,'MKL_NUM_THREADS':threads,
                      'CAD_TRACKING_ROOT':str(cfg.ROOT),'MAX_JOBS':str(cfg.SETTINGS['runtime']['build_jobs']),'CAD_TRACKING_PYTHON':str(python),'CAD_TRACKING_SAM6D':str(cfg.SAM6D),'CAD_TRACKING_NATIVE':str(cfg.NATIVE),
                      'SAM6D_CHECKPOINT_CACHE':str(cfg.path('checkpoints')),
                      'FOUNDATIONPOSE_PYTHON':str(cfg.path('foundationpose_python')),'FOUNDATIONPOSE_ROOT':str(cfg.path('foundationpose')),
                      'FOUNDATIONPOSE_NATIVE':str(cfg.NATIVE/'foundationpose'),
                      'MPLCONFIGDIR':str(cfg.path('matplotlib_cache')),'XDG_CACHE_HOME':str(cfg.path('runtime_cache')),
                      'TORCH_EXTENSIONS_DIR':str(cfg.path('runtime_cache')/'torch_extensions')}.items(): env.setdefault(key,value)
    routes = {'ui':cfg.PIPELINE/'sam6d_gui.py','suite':cfg.PIPELINE/'suites/sam6d_suite.py',
              'evaluate':cfg.PIPELINE/'tools/evaluate_identification.py',
              'detect':cfg.PIPELINE/'tools/run_detection.py','scores':cfg.PIPELINE/'tools/score_all_cads.py',
              'prepare':cfg.PIPELINE/'tools/prepare.py','inspect':cfg.PIPELINE/'tools/inspect_data.py',
              'audit':cfg.PIPELINE/'tools/audit_templates.py','sam':cfg.SAM6D/'pddl_pipeline.py'}
    if args.command.startswith('install'):
        command=['bash',str(cfg.PIPELINE/'setup'/('install_models.sh' if args.command=='install' else 'install_foundationpose.sh')),*extra]
    elif args.command=='test':
        command=[str(python),'-B','-m','pytest','-p','no:cacheprovider',str(cfg.PIPELINE/'tests'),str(cfg.ROOT/'alignment_pose_cad/tests'),*extra]
    else:
        command=[str(python),'-B',str(routes[args.command]),*extra]
        if args.command=='sam' and extra and extra[0]=='render':
            command += ['--blender-install-path',str(cfg.path('blender').parents[1]),'--render-size',str(cfg.SETTINGS['runtime']['render_size']),'--render-samples',str(cfg.SETTINGS['runtime']['render_samples'])]
    os.execvpe(command[0],command,env)

if __name__=='__main__': raise SystemExit(main())
