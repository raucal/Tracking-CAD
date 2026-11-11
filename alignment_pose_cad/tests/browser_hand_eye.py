"""Browser QA with simulated devices, temporary settings and no physical action."""
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'pipeline'),str(Path(__file__).resolve().parent)]
from test_http import HTTPTests
import json
from alignment_pose_cad.config import load
from unittest.mock import patch


def main():
    fixture=HTTPTests('test_status_assets_and_same_origin_guard')
    try:
        fixture.setUp()
        rotation='--rotation' in sys.argv
        automatic='--auto' in sys.argv
        trial='--trial' in sys.argv or rotation or automatic
        if trial:
            manifest,parts=fixture.setup_multi_trial()
            fixture.app.assembly_catalogs=lambda:[{'key':'printed_assembly','label':'Test assembly',
                'parts':[{'key':p['key'],'name':p['name'],'path':p['source_stl']} for p in parts]}]
            catalog=patch('sam6d_ui.assembly_identification.assembly_catalog',return_value=(manifest,parts))
            catalog.start();fixture.addCleanup(catalog.stop)
            original=fixture.app._run
            def infer(*args):
                original(*args)
                if args[2].get('auto_assembly'):
                    with fixture.app.lock:fixture.app.jobs[args[0]]['result']['cad_key']='03_option_cassette'
            fixture.app._run=infer
        else:fixture.use_shared_hand_eye(confirmed=True)
        if rotation:
            fixture.cfg['alignment']['orientation_mode']='cad_axes'
            fixture.cfg['motion']['max_rotation_deg']=90
            fixture.app.alignment.settings_file.write_text(json.dumps(fixture.cfg))
            fixture.app.alignment.robot.cfg=load(fixture.app.alignment.settings_file)[0]
        output=Path(tempfile.mkdtemp(prefix='cad-hand-eye-browser-proof-'))
        node=Path(__file__).resolve().parents[3]/'ASAP/dashboard/.sites-runtime/node/bin/node'
        result=subprocess.run([str(node),str(Path(__file__).with_suffix('.mjs')),fixture.url,str(output),
                               'automatic' if automatic else 'rotation' if rotation else 'trial' if trial else 'preview'],timeout=60)
        assert result.returncode==0,'Browser verification failed'
        assert fixture.app.alignment.robot.control.calls==[], 'Unexpected motion in preview'
        print('Proof:',output)
    finally:
        fixture.doCleanups()


if __name__=='__main__':main()
