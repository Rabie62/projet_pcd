"""
Pytest conftest — mocks heavy ML dependencies so tests can import project modules.
"""
import sys
from unittest.mock import MagicMock


def _setup_torch_mock():
    """Create a comprehensive torch mock with all needed submodules."""
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.functional = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.float16 = MagicMock()
    torch_mock.float32 = MagicMock()
    class _NoGradContext:
        def __call__(self, fn=None):
            if callable(fn):
                return fn
            return self
        def __enter__(self):
            return None
        def __exit__(self, *args):
            return False

    torch_mock.no_grad = _NoGradContext()
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


def _setup_monai_mock():
    monai_mock = MagicMock()
    sys.modules["monai"] = monai_mock
    sys.modules["monai.networks"] = MagicMock()
    sys.modules["monai.networks.nets"] = MagicMock()
    sys.modules["monai.transforms"] = MagicMock()


def _setup_scipy_mock():
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


def _setup_sqlalchemy_mock():
    sa_mock = MagicMock()
    sys.modules["sqlalchemy"] = sa_mock
    sys.modules["sqlalchemy.orm"] = MagicMock()
    sys.modules["sqlalchemy.sql"] = MagicMock()
    sys.modules["sqlalchemy.sql.expression"] = MagicMock()


def _setup_requests_mock():
    sys.modules["requests"] = MagicMock()


def _setup_langgraph_mock():
    lg_mock = MagicMock()
    sys.modules["langgraph"] = lg_mock
    sys.modules["langgraph.graph"] = lg_mock.graph
    sys.modules["langchain_core"] = MagicMock()
    sys.modules["langchain_core.messages"] = MagicMock()


def _setup_transformers_mock():
    tf_mock = MagicMock()
    sys.modules["transformers"] = tf_mock


# Set up all mocks before any test imports
_torch = _setup_torch_mock()
_setup_monai_mock()
_setup_scipy_mock()
_setup_sqlalchemy_mock()
_setup_requests_mock()
_setup_langgraph_mock()
_setup_transformers_mock()
sys.modules["loguru"] = MagicMock()
sys.modules["config.settings"] = MagicMock()
