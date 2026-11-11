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


def configure_predictor_weights(predictor, weights):
    """Override only the two pinned constructor paths, preserving upstream source."""
    import inspect
    import textwrap
    original = getattr(predictor, '_cadtracking_original_init', predictor.__init__)
    source = textwrap.dedent(inspect.getsource(original))
    expression = '{code_dir}/../../weights/{self.run_name}'
    if source.count(expression) != 2:
        raise RuntimeError('FoundationPose checkpoint loader differs from the supported revision')
    source = source.replace(expression, '{_cadtracking_weights}/{self.run_name}')
    namespace = dict(original.__globals__, _cadtracking_weights=str(weights))
    exec(compile(source, inspect.getfile(original), 'exec'), namespace)
    predictor._cadtracking_original_init = original
    predictor.__init__ = namespace['__init__']


def configure_weights():
    """Point both complementary FoundationPose predictors at the configured weights."""
    from settings import path
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    for predictor in (ScorePredictor, PoseRefinePredictor):
        configure_predictor_weights(predictor, path('foundationpose_weights'))
