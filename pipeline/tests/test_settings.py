"""Configuration portability, profile inheritance and command precedence."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import ROOT, profile, relative_path


class SettingsTests(unittest.TestCase):
    def test_relative_profile_paths_use_profile_directory(self):
        with tempfile.TemporaryDirectory(prefix='CAD settings ') as folder:
            config=Path(folder)/'interface.json'
            config.write_text(json.dumps({'server':{'port':8871,'capture_dir':'my captures'}}))
            loaded=profile('interface',config)
            self.assertEqual(loaded['server']['capture_dir'],Path(folder)/'my captures')
            self.assertEqual(loaded['server']['port'],8871)
            self.assertEqual(loaded['inference']['stability'],.97)
            self.assertEqual(loaded['server']['data_dir'],ROOT/'results/interface')

    def test_unknown_profile_keys_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'suite.json';config.write_text('{"confidnce":0.1}')
            with self.assertRaisesRegex(ValueError,'unknown'):profile('suite',config)

    def test_relocated_checkout_with_spaces_and_no_old_alias(self):
        with tempfile.TemporaryDirectory(prefix='relocated CAD ') as folder:
            root=Path(folder)/'another name'
            (root/'pipeline').mkdir(parents=True)
            shutil.copy2(ROOT/'pipeline/settings.py',root/'pipeline/settings.py')
            shutil.copy2(ROOT/'pipeline/launcher.py',root/'pipeline/launcher.py')
            shutil.copy2(ROOT/'run.sh',root/'run.sh')
            shutil.copytree(ROOT/'config',root/'config')
            env=os.environ.copy();env.pop('CAD_TRACKING_SETTINGS',None)
            result=subprocess.run(['bash',str(root/'run.sh'),'settings'],cwd=folder,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            values=json.loads(result.stdout)
            self.assertEqual(values['paths']['sam6d'],str(root/'models/SAM-6D'))
            self.assertEqual(values['paths']['foundationpose'],str(root/'models/FoundationPoseOfficial'))
            self.assertNotIn('/home/raca',result.stdout)

    def test_suite_cli_overrides_profile_without_running_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'suite.json';config.write_text(json.dumps({'confidence':.63,'stability_score':.94,'segmentor':'fastsam','frame':['control_board']}))
            result=subprocess.run(['bash',str(ROOT/'run.sh'),'suite','run','--config',str(config),'--confidence','.71','--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--confidence 0.71',result.stdout)
            self.assertIn('--stability-score 0.94',result.stdout)
            self.assertIn('--segmentor fastsam',result.stdout)
            self.assertIn('--frame control_board',result.stdout)


if __name__=='__main__':unittest.main()
