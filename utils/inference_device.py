"""Device selection for the UniFace video pipelines."""

from contextlib import contextmanager
from inspect import signature
import json


PROVIDERS = {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'openvino': 'OpenVINOExecutionProvider',
}


def model_provider_kwargs(model_class, providers):
    """Use the public provider argument on newer UniFace models."""
    if providers is not None and 'providers' in signature(model_class).parameters:
        return {'providers': providers}
    return {}


def resolve_providers(device, openvino_device='CPU', openvino_precision='FP32'):
    """Select a provider explicitly so UniFace can use OpenVINO in auto mode."""
    if device != 'auto' and device not in PROVIDERS:
        raise ValueError(f'Unknown inference device: {device}')
    if openvino_device not in ('CPU', 'GPU'):
        raise ValueError(f'Unknown OpenVINO device: {openvino_device}')
    if openvino_precision not in ('FP32', 'FP16'):
        raise ValueError(f'Unknown OpenVINO precision: {openvino_precision}')
    if device != 'openvino' and (openvino_device != 'CPU' or openvino_precision != 'FP32'):
        raise ValueError('OpenVINO device and precision options require --device openvino.')
    if openvino_precision == 'FP16' and openvino_device != 'GPU':
        raise ValueError('OpenVINO FP16 requires --openvino-device GPU.')
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            'ONNX Runtime is missing or incomplete in this Python environment. '
            'For OpenVINO, repair it with: '
            'env/bin/python -m pip install --force-reinstall --no-deps '
            'onnxruntime-openvino==1.24.1. '
            'Do not install onnxruntime or onnxruntime-gpu into the same environment.'
        ) from error
    if device == 'auto':
        for preferred in ('CoreMLExecutionProvider', 'CUDAExecutionProvider',
                          'OpenVINOExecutionProvider', 'CPUExecutionProvider'):
            if preferred in available:
                return [preferred]
        raise RuntimeError(f'No supported ONNX Runtime provider is available: {available}')
    provider = PROVIDERS[device]
    if provider not in available:
        raise RuntimeError(
            f'{device} requires {provider}, but this ONNX Runtime installation '
            f'only has: {", ".join(available)}')
    if device == 'openvino' and openvino_device == 'GPU':
        config = {'GPU': {'INFERENCE_PRECISION_HINT':
                          'f16' if openvino_precision == 'FP16' else 'f32'}}
        return [(provider, {'device_type': 'GPU',
                            'load_config': json.dumps(config)})]
    return [provider]


@contextmanager
def uniface_providers(providers):
    """Support UniFace 2.0 models, which lack a public providers argument."""
    if providers is None:
        yield
        return
    from uniface.detection import RetinaFace

    if 'providers' in signature(RetinaFace).parameters:
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
    requested = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    for name, model in models.items():
        active = model.session.get_providers()
        if not active or active[0] != requested:
            raise RuntimeError(
                f'{name} requested {providers[0]}, but active providers are {active}. '
                'ONNX Runtime may have fallen back to CPU.')
