"""Device selection for the UniFace 2.0 models used by the video pipelines."""

from contextlib import contextmanager


PROVIDERS = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'openvino': 'OpenVINOExecutionProvider',
}


def resolve_providers(device):
    """Return an explicit ONNX provider list, or None for UniFace auto mode."""
    if device == 'auto':
        return None
    if device not in PROVIDERS:
        raise ValueError(f'Unknown inference device: {device}')
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError('ONNX Runtime is required for face inference.') from error
    provider = PROVIDERS[device]
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(
            f'{device} requires {provider}, but this ONNX Runtime installation '
            f'only has: {", ".join(available)}')
    return [provider]


@contextmanager
def uniface_providers(providers):
    """Scope UniFace 2.0 automatic provider resolution to model construction."""
    if providers is None:
        yield
        return
    from uniface import onnx_utils

    original = onnx_utils.get_available_providers
    onnx_utils.get_available_providers = lambda: providers
    try:
        yield
    finally:
        onnx_utils.get_available_providers = original


def verify_model_providers(models, providers):
    """Catch unexpected CPU fallback before processing video frames."""
    if providers is None:
        return
    requested = providers[0]
    for name, model in models.items():
        active = model.session.get_providers()
        if not active or active[0] != requested:
            raise RuntimeError(
                f'{name} requested {requested}, but active providers are {active}.')
