"""Coverage for the rate33 half-rate channel-frame submodule."""

import numpy as np
import pytest

import blip25_mbe
from blip25_mbe import rate33 as r33


def _sample_b(seed: int) -> list[int]:
    """A deterministic, width-legal b̂₀..b̂₈ parameter array."""
    state = seed & 0xFFFFFFFF
    b = []
    for w in r33.AMBE_PARAM_WIDTHS:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        b.append(state & ((1 << w) - 1))
    return b


def test_module_is_reexported_namespace() -> None:
    assert blip25_mbe.rate33 is r33
    assert list(r33.INFO_WIDTHS) == [12, 12, 11, 14]
    assert list(r33.AMBE_VECTOR_WIDTHS) == [12, 12, 11, 14]
    assert list(r33.AMBE_PARAM_WIDTHS) == [7, 5, 5, 9, 7, 5, 4, 4, 3]
    assert sum(r33.AMBE_PARAM_WIDTHS) == 49
    assert sum(r33.INFO_WIDTHS) == r33.INFO_BITS_TOTAL == 49
    assert r33.AMBE_B_COUNT == 9
    assert r33.DIBITS_PER_FRAME == 36
    assert r33.SOFT_BITS == 72
    assert len(r33.R34_BIT_ORDER) == 49
    assert sorted(r33.R34_BIT_ORDER) == list(range(49))  # a bijection


@pytest.mark.parametrize("seed", [1, 0xDEADBEEF, 0xCAFEBABE, 0xA5A5A5A5, 42])
def test_prioritize_deprioritize_roundtrip(seed: int) -> None:
    b = _sample_b(seed)
    u = r33.prioritize(b)
    assert len(u) == 4
    # Each info vector stays within its declared width.
    for v, w in zip(u, r33.INFO_WIDTHS):
        assert v >> w == 0
    assert r33.deprioritize(u) == b


@pytest.mark.parametrize("seed", [1, 0xDEADBEEF, 7, 0x5F5F5F5F])
def test_no_fec_byte_orders_roundtrip(seed: int) -> None:
    u = r33.prioritize(_sample_b(seed))

    r34 = r33.pack_no_fec(u)
    assert isinstance(r34, bytes) and len(r34) == 7
    assert r33.unpack_no_fec(r34) == list(u)

    natural = r33.info_to_natural(u)
    assert isinstance(natural, bytes) and len(natural) == 7
    assert r33.natural_to_info(natural) == list(u)

    # The two 7-byte layouts are genuinely different (for non-trivial u).
    if any(u):
        assert r34 != natural


@pytest.mark.parametrize("seed", [1, 0xDEADBEEF, 99, 0x1234ABCD])
def test_encode_decode_frame_roundtrip(seed: int) -> None:
    u = r33.prioritize(_sample_b(seed))
    dibits = r33.encode_frame(u)
    assert len(dibits) == r33.DIBITS_PER_FRAME
    assert all(0 <= d <= 3 for d in dibits)

    frame = r33.decode_frame(dibits)
    assert frame.info == list(u)
    assert frame.errors == [0, 0, 0, 0]
    assert frame.error_total() == 0

    # decode_frame == deinterleave + decode_code_vectors.
    assert r33.decode_code_vectors(r33.deinterleave(dibits)).info == list(u)


@pytest.mark.parametrize("seed", [3, 0xBEEF, 0x0F0F0F0F])
def test_byte_dibit_roundtrip(seed: int) -> None:
    dibits = r33.encode_frame(r33.prioritize(_sample_b(seed)))
    packed = r33.pack_dibits(dibits)
    assert isinstance(packed, bytes) and len(packed) == 9  # 36 dibits / 4
    assert r33.unpack_dibits(packed) == list(dibits)


def test_pack_dibits_rejects_ragged_input() -> None:
    with pytest.raises(ValueError):
        r33.pack_dibits([0, 1, 2])  # not a multiple of 4


@pytest.mark.parametrize(
    "fn", [r33.unpack_no_fec, r33.natural_to_info, r33.fields_from_no_fec]
)
def test_seven_byte_helpers_reject_wrong_length(fn) -> None:
    with pytest.raises(ValueError):
        fn(b"\x00" * 9)


def test_fields_from_fec_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        r33.fields_from_fec(b"\x00" * 7)


def test_u0_range_checked() -> None:
    for fn in (r33.pn_sequence, r33.modulation_masks):
        assert len(fn(0)) in (24, 4)
        with pytest.raises(ValueError):
            fn(4096)


def test_soft_decode_matches_hard() -> None:
    u = r33.prioritize(_sample_b(0xC0FFEE))
    dibits = r33.encode_frame(u)
    soft = np.empty(r33.SOFT_BITS, dtype=np.int8)
    for i, d in enumerate(dibits):
        soft[2 * i] = 50 if (d >> 1) & 1 else -50
        soft[2 * i + 1] = 50 if d & 1 else -50
    frame = r33.decode_frame_soft(soft)
    assert frame.info == list(u)
    assert frame.error_total() == 0


def test_soft_decode_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        r33.decode_frame_soft(np.zeros(71, dtype=np.int8))


def test_field_dump_from_real_vocoder_frames() -> None:
    # A real AMBE+2 wire frame's parameter fields are recoverable, and
    # the FEC (9-byte) and r34 no-FEC (7-byte) forms carry the same info
    # layer, so they deprioritize to identical fields.
    pcm = (2000 * np.sin(2 * np.pi * 440 * np.arange(160) / 8000)).astype(np.int16)

    vc_fec = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)
    vc_nofec = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_2450X2450)
    frame9 = vc_fec.encode_pcm(pcm)
    frame7 = vc_nofec.encode_pcm(pcm)
    assert len(frame9) == 9 and len(frame7) == 7

    b_fec = r33.fields_from_fec(frame9)
    b_nofec = r33.fields_from_no_fec(frame7)
    assert len(b_fec) == 9
    assert b_fec == b_nofec

    # The decoded FEC frame is error-free on a clean self-encode.
    frame = r33.decode_frame(r33.unpack_dibits(frame9))
    assert frame.error_total() == 0
