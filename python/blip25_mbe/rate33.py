"""Half-rate (DVSI rate-33) AMBE+2 channel-frame toolkit.

The bit-field layer *below* the PCM↔wire façade: the four info vectors
``û₀..û₃``, the FEC code vectors ``c₀..c₃``, the deprioritized
``b̂₀..b̂₈`` voice-parameter fields, and the r34 / natural (AMBE_d)
no-FEC byte orders. Re-exports the compiled ``_blip25_mbe.rate33``
submodule symbols into the stable ``blip25_mbe.rate33`` namespace.

The one-call field extractors cover all three wire forms::

    from blip25_mbe import rate33
    b = rate33.fields_from_fec(frame9)       # 9-byte FEC frame
    b = rate33.fields_from_no_fec(frame7)    # 7-byte DVSI r34 no-FEC
    b = rate33.fields_from_natural(frame7)   # 7-byte natural / AMBE_d
"""

from ._blip25_mbe.rate33 import (
    AMBE_B_COUNT,
    AMBE_PARAM_WIDTHS,
    AMBE_VECTOR_WIDTHS,
    CODE_WIDTHS,
    DIBITS_PER_FRAME,
    INFO_BITS_TOTAL,
    INFO_WIDTHS,
    PN_SEQ_LEN,
    R34_BIT_ORDER,
    SOFT_BITS,
    Rate33Frame,
    decode_code_vectors,
    decode_frame,
    decode_frame_soft,
    deinterleave,
    demodulate,
    deprioritize,
    encode_frame,
    fields_from_fec,
    fields_from_natural,
    fields_from_no_fec,
    info_to_natural,
    interleave,
    modulation_masks,
    natural_to_info,
    pack_dibits,
    pack_no_fec,
    pn_sequence,
    prioritize,
    unpack_dibits,
    unpack_no_fec,
)

__all__ = [
    "AMBE_B_COUNT",
    "AMBE_PARAM_WIDTHS",
    "AMBE_VECTOR_WIDTHS",
    "CODE_WIDTHS",
    "DIBITS_PER_FRAME",
    "INFO_BITS_TOTAL",
    "INFO_WIDTHS",
    "PN_SEQ_LEN",
    "R34_BIT_ORDER",
    "SOFT_BITS",
    "Rate33Frame",
    "decode_code_vectors",
    "decode_frame",
    "decode_frame_soft",
    "deinterleave",
    "demodulate",
    "deprioritize",
    "encode_frame",
    "fields_from_fec",
    "fields_from_natural",
    "fields_from_no_fec",
    "info_to_natural",
    "interleave",
    "modulation_masks",
    "natural_to_info",
    "pack_dibits",
    "pack_no_fec",
    "pn_sequence",
    "prioritize",
    "unpack_dibits",
    "unpack_no_fec",
]
