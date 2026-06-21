"""Type stubs for the DVSI soft-decision submodule."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

SD_HEADER: int
SD_PACKET_WORDS: int
SD_OVERHEAD_WORDS: int
SD_DATA_WORDS: int
SD_SLOTS: int
RATE_33_CHANNEL_BITS: int
IMBE_FULL_RATE_CHANNEL_BITS: int
DVSI_P25_FULLRATE_FEC: List[int]
DVSI_P25_FULLRATE_NOFEC: List[int]

class SdPacketHeader:
    power_control: int
    control_word1: int
    rate_info: List[int]
    dtmf_control: int
    control_word2: int

    def __init__(
        self,
        rate_info: Sequence[int] = ...,
        *,
        power_control: int = ...,
        control_word1: int = ...,
        dtmf_control: int = ...,
        control_word2: int = ...,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...

def llr_to_sd_nibble(llr: int) -> int: ...
def sd_nibble_to_llr(n: int) -> int: ...
def pack_channel_bits(
    channel_llrs: NDArray[np.int8], header: SdPacketHeader
) -> NDArray[np.uint16]: ...
def unpack_packet(
    words: NDArray[np.uint16],
) -> Tuple[SdPacketHeader, NDArray[np.int8]]: ...
def packet_to_bytes(words: NDArray[np.uint16]) -> bytes: ...
def pack_nibble_stream(channel_llrs: NDArray[np.int8]) -> bytes: ...
def unpack_nibble_stream(data: bytes) -> NDArray[np.int8]: ...
