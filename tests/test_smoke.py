"""Sanity round-trip across all four rates."""

import numpy as np
import pytest

import blip25_mbe


ALL_RATES = [
    blip25_mbe.Rate.IMBE_7200X4400,
    blip25_mbe.Rate.IMBE_4400X4400,
    blip25_mbe.Rate.AMBEPLUS2_3600X2450,
    blip25_mbe.Rate.AMBEPLUS2_2450X2450,
]


@pytest.mark.parametrize("rate", ALL_RATES)
def test_one_frame_round_trip(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    pcm_in = np.zeros(vc.frame_samples, dtype=np.int16)

    bits = vc.encode_pcm(pcm_in)
    assert isinstance(bits, bytes)
    assert len(bits) == vc.fec_frame_bytes

    pcm_out = vc.decode_bits(bits)
    assert pcm_out.dtype == np.int16
    assert pcm_out.shape == (vc.frame_samples,)


def test_live_encoder_chunks_arbitrary_sizes() -> None:
    enc = blip25_mbe.LiveEncoder(blip25_mbe.Rate.IMBE_7200X4400)
    # Push 500 samples in 73-sample chunks (deliberately not a frame
    # multiple). We should get floor(500 / 160) = 3 frames out by the
    # time we've pushed all 500.
    rng = np.random.default_rng(0)
    total_pcm = rng.integers(-1000, 1000, size=500, dtype=np.int16)
    frames_out: list[bytes] = []
    for i in range(0, len(total_pcm), 73):
        chunk = total_pcm[i : i + 73]
        frames_out.extend(enc.push(chunk))
    assert len(frames_out) == 3
    assert all(len(f) == 18 for f in frames_out)


def test_live_decoder_round_trip() -> None:
    rate = blip25_mbe.Rate.AMBEPLUS2_3600X2450
    vc = blip25_mbe.Vocoder(rate)
    silent = np.zeros(vc.frame_samples, dtype=np.int16)
    bits = b"".join(vc.encode_pcm(silent) for _ in range(4))

    dec = blip25_mbe.LiveDecoder(rate)
    pcm_frames = dec.push(bits)
    assert len(pcm_frames) == 4
    assert all(f.shape == (vc.frame_samples,) and f.dtype == np.int16 for f in pcm_frames)


def test_transcoder_imbe_to_ambe_plus2() -> None:
    src = blip25_mbe.Rate.IMBE_7200X4400
    dst = blip25_mbe.Rate.AMBEPLUS2_3600X2450
    enc = blip25_mbe.Vocoder(src)
    tc = blip25_mbe.Transcoder(src, dst)

    silent = np.zeros(enc.frame_samples, dtype=np.int16)
    src_bits = enc.encode_pcm(silent)
    dst_bits = tc.transcode(src_bits)
    assert len(dst_bits) == 9  # AMBE+2 FEC frame is 9 bytes
