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
