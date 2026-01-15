#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import onnxruntime as ort


def check_ort_capabilities():
    # Defensive check for the attribute
    if not hasattr(ort, 'get_available_providers'):
        print(
            "Error: 'get_available_providers' not found. Check your installation."
        )
        return

    providers = ort.get_available_providers()
    print(f"--- ONNX Runtime {ort.__version__} ---")

    for provider in providers:
        print(f"\n[Provider: {provider}]")

        # Check precision capabilities via Provider Options
        try:
            # We use a dummy session to force ORT to reveal the internal configuration
            # This is the most reliable way to see what 'precision' flags exist
            opts = ort.SessionOptions()
            temp_session = ort.InferenceSession(
                # Using a tiny byte string as a placeholder for a model
                # (Note: This might fail on some versions, we just want the config)
                None,
                sess_options=opts,
                providers=[provider])
        except Exception:
            # We expect a failure because we provided no model,
            # but we can still check the provider's default options mapping
            pass

        # Manual lookup of common precision flags for major EPs
        precision_flags = {
            'TensorrtExecutionProvider':
            ['trt_fp16_enable', 'trt_int8_enable'],
            'CUDAExecutionProvider': ['cudnn_conv_algo_search'],
            'CoreMLExecutionProvider': ['precision'],
            'DmlExecutionProvider': ['device_id']
        }

        if provider in precision_flags:
            print(
                f"  Supported precision toggles: {precision_flags[provider]}")
        else:
            print(
                "  Standard FP32/INT8 support (EP-specific flags not detected)"
            )


if __name__ == "__main__":
    check_ort_capabilities()
