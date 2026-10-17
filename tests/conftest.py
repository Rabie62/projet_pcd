"""
Pytest conftest — mocks heavy ML dependencies so tests can import project modules.
"""
import sys
from unittest.mock import MagicMock


def setup_torch():
    """Create a comprehensive torch mock with all needed submodules."""
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.functional = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.float16 = MagicMock()
    torch_mock.float32 = MagicMock()
    class NoGradContext:
        def __call__(self, fn=None):
            if callable(fn):
                return fn
            return self
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False

    torch_mock.no_grad = NoGradContext()
    torch_mock.Tensor = MagicMock
    torch_mock.from_numpy = MagicMock()
    torch_mock.softmax = MagicMock()
    torch_mock.argmax = MagicMock()
    torch_mock.load = MagicMock()

    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = torch_mock.nn
    sys.modules["torch.nn.functional"] = torch_mock.nn.functional
    sys.modules["torch.cuda"] = torch_mock.cuda
    sys.modules["torchvision"] = MagicMock()
    sys.modules["torchvision.models"] = MagicMock()
    return torch_mock


def setup_monai():
    monai_mock = MagicMock()
    sys.modules["monai"] = monai_mock
    sys.modules["monai.networks"] = MagicMock()
    sys.modules["monai.networks.nets"] = MagicMock()
    sys.modules["monai.transforms"] = MagicMock()


def setup_scipy():
    try:
        import scipy.ndimage
        import scipy.spatial.distance
        return
    except ImportError:
        pass
    scipy_mock = MagicMock()
    sys.modules["scipy"] = scipy_mock
    sys.modules["scipy.spatial"] = MagicMock()
    sys.modules["scipy.spatial.distance"] = MagicMock()
    sys.modules["scipy.ndimage"] = MagicMock()


def setup_sqlalchemy():
    sa_mock = MagicMock()
    sys.modules["sqlalchemy"] = sa_mock
    sys.modules["sqlalchemy.orm"] = MagicMock()
    sys.modules["sqlalchemy.sql"] = MagicMock()
    sys.modules["sqlalchemy.sql.expression"] = MagicMock()


def setup_requests():
    sys.modules["requests"] = MagicMock()


def setup_langgraph():
    lg_mock = MagicMock()
    sys.modules["langgraph"] = lg_mock
    sys.modules["langgraph.graph"] = lg_mock.graph
    sys.modules["langchain_core"] = MagicMock()
    sys.modules["langchain_core.messages"] = MagicMock()


def setup_transformers():
    tf_mock = MagicMock()
    sys.modules["transformers"] = tf_mock


def setup_smp():
    smp_mock = MagicMock()
    sys.modules["segmentation_models_pytorch"] = smp_mock


# Set up all mocks before any test imports
mock_torch = setup_torch()
setup_monai()
setup_smp()
setup_scipy()
setup_sqlalchemy()
setup_requests()
setup_langgraph()
setup_transformers()
sys.modules["loguru"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
