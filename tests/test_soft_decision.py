"""Coverage for the DVSI soft-decision (LLR) packet submodule."""

import numpy as np
import pytest

import blip25_mbe
from blip25_mbe import dvsi_soft_decision as dsd


def test_module_is_reexported_namespace() -> None:
    # Reachable both as a package submodule and as the compiled child.
    assert blip25_mbe.dvsi_soft_decision is dsd
    assert dsd.SD_HEADER == 0x13EC
    assert dsd.SD_PACKET_WORDS == 60
    assert dsd.SD_SLOTS == 192
    assert dsd.RATE_33_CHANNEL_BITS == 72
    assert dsd.IMBE_FULL_RATE_CHANNEL_BITS == 144


def test_llr_nibble_roundtrip_every_value() -> None:
    # The app-note anchor rows.
    assert dsd.llr_to_sd_nibble(-127) == 0b0000  # most confident 0
    assert dsd.llr_to_sd_nibble(0) == 0b0111  # llr 0 is hard-0
    assert dsd.llr_to_sd_nibble(1) == 0b1000  # least confident 1
    assert dsd.llr_to_sd_nibble(127) == 0b1111  # most confident 1
    # Every 4-bit SD value round-trips through a representative LLR.
    for n in range(16):
        assert dsd.llr_to_sd_nibble(dsd.sd_nibble_to_llr(n)) == n


def test_pack_unpack_roundtrip_preserves_header_and_sd_field() -> None:
    header = dsd.SdPacketHeader(
        rate_info=[0x0558, 0x086B, 0x1030, 0x0000, 0x0190],
        power_control=0xAB,
        control_word1=0xCD,
        dtmf_control=0x1234,
        control_word2=0x5678,
    )
    llrs = np.full(dsd.RATE_33_CHANNEL_BITS, 100, dtype=np.int8)

    pkt = dsd.pack_channel_bits(llrs, header)
    assert pkt.dtype == np.uint16
    assert len(pkt) == dsd.SD_PACKET_WORDS
    assert int(pkt[0]) == dsd.SD_HEADER

    out_header, sd = dsd.unpack_packet(pkt)
    assert out_header == header
    assert len(sd) == dsd.SD_SLOTS
    # Active Rate-33 slots are hard-1; the rest fill most-confident-0.
    assert np.all(sd[: dsd.RATE_33_CHANNEL_BITS] > 0)
    assert np.all(np.vectorize(dsd.llr_to_sd_nibble)(sd[dsd.RATE_33_CHANNEL_BITS:]) == 0)


def test_sd0_is_high_nibble_of_first_data_word() -> None:
    # SD0 = most-confident-1 (0xF), SD1..3 = most-confident-0 (0x0).
    llrs = np.array([127, -127, -127, -127], dtype=np.int8)
    pkt = dsd.pack_channel_bits(llrs, dsd.SdPacketHeader())
    assert int(pkt[dsd.SD_OVERHEAD_WORDS]) == 0xF000


def test_packet_to_bytes_big_endian() -> None:
    pkt = dsd.pack_channel_bits(np.zeros(0, dtype=np.int8), dsd.SdPacketHeader())
    wire = dsd.packet_to_bytes(pkt)
    assert isinstance(wire, bytes)
    assert len(wire) == dsd.SD_PACKET_WORDS * 2
    assert wire[:2] == bytes([0x13, 0xEC])  # header high byte first


def test_nibble_stream_roundtrip() -> None:
    llrs = np.array([127, -127, 50, -10, 0, 1], dtype=np.int8)
    stream = dsd.pack_nibble_stream(llrs)
    assert isinstance(stream, bytes)
    assert len(stream) == 3  # two nibbles per byte
    back = dsd.unpack_nibble_stream(stream)
    # Lossy (4-bit) but the SD-nibble of each value round-trips exactly.
    assert np.array_equal(
        np.vectorize(dsd.llr_to_sd_nibble)(llrs),
        np.vectorize(dsd.llr_to_sd_nibble)(back),
    )


def test_too_many_channel_bits_raises() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        dsd.pack_channel_bits(np.zeros(193, dtype=np.int8), dsd.SdPacketHeader())


def test_unpack_rejects_wrong_word_count() -> None:
    with pytest.raises(ValueError, match="packet words"):
        dsd.unpack_packet(np.zeros(10, dtype=np.uint16))


def test_unpack_rejects_bad_header() -> None:
    bad = np.zeros(dsd.SD_PACKET_WORDS, dtype=np.uint16)  # word 0 != 0x13EC
    with pytest.raises(ValueError, match="bad SD packet header"):
        dsd.unpack_packet(bad)


def test_header_defaults_and_field_mutation() -> None:
    h = dsd.SdPacketHeader()
    assert list(h.rate_info) == [0, 0, 0, 0, 0]
    assert h.power_control == 0
    h.power_control = 0x42
    h.rate_info = [1, 2, 3, 4, 5]
    assert h.power_control == 0x42
    assert list(h.rate_info) == [1, 2, 3, 4, 5]
