"""Bridge to SDU compatibility kept outside the official model submodule."""
import importlib.util
import sys
from settings import ROOT


def install(official):
    name = '_cadtracking_foundationpose_sdu_runtime'
    if name not in sys.modules:
        filename = ROOT / 'integrations/foundationpose_sdu/runtime.py'
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].install(official, ROOT / '.cache/foundationpose_sdu')
