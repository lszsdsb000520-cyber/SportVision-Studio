import av
import cv2
import numpy as np
import pytest

from pingpong.fast_input import ModelInputSampler, sampler


@pytest.mark.skipif(sampler() is None,reason='Native model-input accelerator is unavailable')
def test_sampled_yuv_pixels_are_bit_identical_to_full_frame_resize():
    generator = np.random.default_rng(47)
    decoder = ModelInputSampler()
    for _ in range(4):
        frame = av.VideoFrame(1920,1080,'yuv420p')
        for plane in frame.planes:
            np.frombuffer(plane,np.uint8)[:] = generator.integers(0,256,plane.buffer_size,dtype=np.uint8)
        expected = cv2.resize(frame.to_ndarray(format='bgr24'),(512,288))
        assert np.array_equal(decoder.resize(frame,0),expected)
        assert decoder.enabled
