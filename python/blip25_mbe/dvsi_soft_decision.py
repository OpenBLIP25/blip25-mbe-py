"""DVSI soft-decision chip handoff — the 4-bit soft-decision (LLR) packet
format for soft-FEC interchange with DVSI AMBE-2000/2020/3000 hardware.

Re-exports the compiled ``_blip25_mbe.dvsi_soft_decision`` submodule symbols
into the stable ``blip25_mbe.dvsi_soft_decision`` namespace.
"""

from ._blip25_mbe.dvsi_soft_decision import (
    DVSI_P25_FULLRATE_FEC,
    DVSI_P25_FULLRATE_NOFEC,
    IMBE_FULL_RATE_CHANNEL_BITS,
    RATE_33_CHANNEL_BITS,
    SD_DATA_WORDS,
    SD_HEADER,
    SD_OVERHEAD_WORDS,
    SD_PACKET_WORDS,
    SD_SLOTS,
    SdPacketHeader,
    llr_to_sd_nibble,
    pack_channel_bits,
    pack_nibble_stream,
    packet_to_bytes,
    sd_nibble_to_llr,
    unpack_nibble_stream,
    unpack_packet,
)

__all__ = [
    "DVSI_P25_FULLRATE_FEC",
    "DVSI_P25_FULLRATE_NOFEC",
    "IMBE_FULL_RATE_CHANNEL_BITS",
    "RATE_33_CHANNEL_BITS",
    "SD_DATA_WORDS",
    "SD_HEADER",
    "SD_OVERHEAD_WORDS",
    "SD_PACKET_WORDS",
    "SD_SLOTS",
    "SdPacketHeader",
    "llr_to_sd_nibble",
    "pack_channel_bits",
    "pack_nibble_stream",
    "packet_to_bytes",
    "sd_nibble_to_llr",
    "unpack_nibble_stream",
    "unpack_packet",
]
