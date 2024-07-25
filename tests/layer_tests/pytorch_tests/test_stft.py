# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSTFT(PytorchLayerTest):
    def _prepare_input(self, win_length, signal_length, out=False, out_dtype="float32"):
        import numpy as np

        np.random.seed(seed=2)
        signal = np.random.randn(1, signal_length).astype(out_dtype)
        window = np.hanning(win_length).reshape([win_length])

        return (np.array(signal, dtype=out_dtype), window.astype(out_dtype))

    def create_model(self, n_fft, hop_length, win_length):
        import torch

        class aten_stft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length):
                super(aten_stft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length

            def forward(self, x, window):
                return torch.stft(
                    x,
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=False,
                    pad_mode="reflect",
                    normalized=False,
                    onesided=True,
                    return_complex=False,
                )

        ref_net = None

        model = aten_stft(n_fft, hop_length, win_length)
        return model, ref_net, "aten::stft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.parametrize(("n_fft", "hop_length", "win_length", "signal_length"), [
        [16, 16 // 4, 16, 48],
        [16, 16 // 2, 8, 48],
        [8, 8 // 4, 5, 48],
        [9, 9 // 4, 7, 48],
        [256, 256 // 8, 256, 1000],
        # [256, 256 // 4, 256, 48], # Invalid n_fft, expect to throw
    ])
    def test_stft(self, n_fft, hop_length, win_length, signal_length, ie_device, precision, ir_version, trace_model):
        self._test(*self.create_model(n_fft, hop_length, win_length), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": win_length, "signal_length": signal_length}, trace_model=trace_model)
