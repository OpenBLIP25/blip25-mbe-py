# blip25-mbe (Python)

Python bindings for [`blip25-mbe`](https://github.com/openBLIP25/blip25-mbe),
a research-grade Rust implementation of the MBE / IMBE / AMBE+2 vocoder
family for P25 Phase 1 and Phase 2 pipelines.

> **Patents.** AMBE+2 is covered by US 8,359,197 until 2028-05-20. See
> [`PATENT_NOTICE.md`](./PATENT_NOTICE.md).

## Install

```bash
pip install blip25-mbe
```

Wheels are published for Linux (manylinux 2_28), macOS (x86_64 + arm64),
and Windows for Python 3.9+.

## Quick start

```python
import numpy as np
import blip25_mbe

# One-shot frame round-trip
vc = blip25_mbe.Vocoder(blip25_mbe.Rate.IMBE_7200X4400)
pcm_in = np.zeros(vc.frame_samples, dtype=np.int16)   # 160 samples = 20 ms
bits = vc.encode_pcm(pcm_in)                          # 18-byte FEC frame
pcm_out = vc.decode_bits(bits)                        # back to np.int16

# Streaming over arbitrary-sized chunks
enc = blip25_mbe.LiveEncoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)
for chunk in pcm_chunks:                              # any size, any rate
    for frame in enc.push(chunk):
        socket.send(frame)                            # 9 bytes each

# Wire-format bridge (Phase 1 IMBE ⇄ Phase 2 AMBE+2)
tc = blip25_mbe.Transcoder(
    blip25_mbe.Rate.IMBE_7200X4400,
    blip25_mbe.Rate.AMBEPLUS2_3600X2450,
)
out_bits = tc.transcode(in_bits)
```

`encode_pcm` accepts any `np.int16` array; `decode_bits` returns a fresh
`np.int16` array. Length must match `vc.frame_samples` / `vc.fec_frame_bytes`.

## Rates

| Python                            | Codec       | Wire size | Bitrate     |
|-----------------------------------|-------------|-----------|-------------|
| `Rate.IMBE_7200X4400`             | IMBE        | 18 bytes  | 7 200 bps   |
| `Rate.IMBE_4400X4400`             | IMBE no-FEC | 11 bytes  | 4 400 bps   |
| `Rate.AMBEPLUS2_3600X2450`        | AMBE+2      |  9 bytes  | 3 600 bps   |
| `Rate.AMBEPLUS2_2450X2450`        | AMBE+2 no-FEC | 7 bytes | 2 450 bps   |

> **`AMBEPLUS2_2450X2450` byte order.** The 7-byte no-FEC frame packs the
> 49 info bits in DVSI **r34 column-interleave** order (byte-exact with
> DVSI's chip rate-index 34 no-FEC stream), *not* naive MSB-first
> sequential. Consumers expecting natural / "AMBE_d" order (mbelib,
> IDAS/NXDN over-the-air) must de-interleave first.

## Encoder tuning (blip25-mbe 0.2.0)

`Vocoder` exposes the upstream encode-quality and denoiser levers. The
AMBE+2 production encode-quality stack (octave-escape pitch guard,
parabolic sub-sample pitch, `M(ξ)` voicing relaxation) is **on by
default** for AMBE+2 rates; full-rate IMBE stays bit-for-bit
spec-faithful. The denoiser front-ends are opt-in (default off).

```python
vc = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)

# Encode-quality stack (per-lever opt-out)
vc.pitch_decide_escape          # -> True   (set_pitch_decide_escape)
vc.pitch_subsample              # -> True   (set_pitch_subsample)
vc.vuv_mxi_grade                # -> True   (set_vuv_mxi_grade)
vc.pitch_refine                 # -> False  §0.4 refine bypassed by the stack
vc.set_vuv_pitch_coef(0.0)      # write-only loudness/shape levers
vc.set_level_scale(True)

# Pre-analysis denoiser front-ends (opt-in 'exceed' levers for noisy audio)
vc.set_denoise(True)                              # log-MMSE STFT denoiser
vc.set_denoise_kind(blip25_mbe.DenoiseKind.WIENER)  # or LOG_MMSE / SPEC_SUB
vc.set_hum_notch(True)                            # 60/120 Hz (US) mains notch
vc.set_hum_notch_mains(50.0)                      # 50/100 Hz (EU)
```

> **Note.** As of 0.2.0, input-side `spectral_subtraction` defaults
> **off** (was on in 0.1.x); enable it with `set_spectral_subtraction(True)`.
> The post-decode `EnhancementMode.CLASSICAL` chain remains on by default.

## DVSI soft-decision packets

`blip25_mbe.dvsi_soft_decision` provides the 4-bit soft-decision (LLR)
packet format for soft-FEC interchange with DVSI AMBE-2000/2020/3000
hardware: `pack_channel_bits` / `unpack_packet`, the raw USB-3000 nibble
stream (`pack_nibble_stream` / `unpack_nibble_stream`), `SdPacketHeader`,
and the verified P25 rate-control vectors.

```python
import numpy as np
from blip25_mbe import dvsi_soft_decision as dsd

llrs = np.array(channel_soft_bits, dtype=np.int8)   # one i8 LLR per bit
header = dsd.SdPacketHeader()       # rate_info words are chip-/rate-specific
packet = dsd.pack_channel_bits(llrs, header)        # uint16[60]
wire = dsd.packet_to_bytes(packet)                  # 120 big-endian bytes
```

> `DVSI_P25_FULLRATE_FEC` / `DVSI_P25_FULLRATE_NOFEC` are the 6-word
> chip `-r` rate-control vectors — a *different* framing from the
> packet's 5-word `rate_info` field; don't cross-assign them.

## Building from source

```bash
pip install maturin
maturin develop --release       # editable install into current venv
pytest                          # smoke tests
```

## Sibling packages

`blip25-mbe` is the vocoder wrapper. Future blip25 components (decoder /
SDR layer) will ship as separate PyPI packages with their own
`blip25_*` import names — they don't share a namespace with this one.

## License

MIT — see [`LICENSE`](./LICENSE).
