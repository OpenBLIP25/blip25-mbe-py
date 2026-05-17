//! Python bindings for `blip25-mbe`.
//!
//! Exposes the chip-shaped façade — `Rate`, `Vocoder`, `Transcoder`,
//! `LiveEncoder`, `LiveDecoder` — with zero-copy numpy interop on the
//! PCM boundary. Tier-1 surface only; setters, the builder, and
//! parameter-domain entry points will land in a follow-on.

use blip25_mbe::vocoder::{self as bv, Rate as BRate};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Map a `VocoderError` to a Python `ValueError` with the upstream
/// `Display` text — preserves the actionable message without
/// inventing a new exception hierarchy.
fn map_err(e: bv::VocoderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// On-wire rate / codec selection. The four variants cover the
/// production P25 storage and transport formats. See the upstream
/// `docs/wire_formats_and_storage.md` for the trade-offs.
#[pyclass(eq, eq_int, frozen, name = "Rate")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRate {
    /// P25 Phase 1 FDMA full-rate IMBE (18-byte FEC frame, 7 200 bps).
    #[pyo3(name = "IMBE_7200X4400")]
    Imbe7200x4400,
    /// IMBE info-only (11-byte frame, 4 400 bps).
    #[pyo3(name = "IMBE_4400X4400")]
    Imbe4400x4400,
    /// P25 Phase 2 TDMA AMBE+2 (9-byte FEC frame, 3 600 bps).
    #[pyo3(name = "AMBEPLUS2_3600X2450")]
    AmbePlus2_3600x2450,
    /// AMBE+2 info-only (7-byte frame, 2 450 bps).
    #[pyo3(name = "AMBEPLUS2_2450X2450")]
    AmbePlus2_2450x2450,
}

impl From<PyRate> for BRate {
    fn from(r: PyRate) -> Self {
        match r {
            PyRate::Imbe7200x4400 => BRate::Imbe7200x4400,
            PyRate::Imbe4400x4400 => BRate::Imbe4400x4400,
            PyRate::AmbePlus2_3600x2450 => BRate::AmbePlus2_3600x2450,
            PyRate::AmbePlus2_2450x2450 => BRate::AmbePlus2_2450x2450,
        }
    }
}

impl From<BRate> for PyRate {
    fn from(r: BRate) -> Self {
        match r {
            BRate::Imbe7200x4400 => PyRate::Imbe7200x4400,
            BRate::Imbe4400x4400 => PyRate::Imbe4400x4400,
            BRate::AmbePlus2_3600x2450 => PyRate::AmbePlus2_3600x2450,
            BRate::AmbePlus2_2450x2450 => PyRate::AmbePlus2_2450x2450,
            // `Rate` is `#[non_exhaustive]` upstream; if a new variant
            // appears, rebuild this crate against the newer
            // `blip25-mbe` to surface it as a `PyRate` variant too.
            other => panic!("blip25-py: unmapped upstream Rate variant {other:?}"),
        }
    }
}

/// One vocoder channel. Stateful across frames (analysis history,
/// LCG state, decoder cross-frame memory). Use [`Vocoder.reset`] to
/// clear between independent streams.
#[pyclass(name = "Vocoder", unsendable)]
pub struct PyVocoder {
    inner: bv::Vocoder,
}

#[pymethods]
impl PyVocoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::Vocoder::new(rate.into()) }
    }

    /// The rate this vocoder was constructed for.
    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    /// PCM samples in one frame (`160` for all current rates).
    #[getter]
    fn frame_samples(&self) -> usize {
        self.inner.frame_samples()
    }

    /// On-wire byte count for one frame at this rate.
    #[getter]
    fn fec_frame_bytes(&self) -> usize {
        self.inner.fec_frame_bytes()
    }

    /// Encode one PCM frame to its on-wire bytes. `pcm` must be a
    /// 1-D `np.int16` array of length [`frame_samples`].
    fn encode_pcm<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.encode_pcm(pcm.as_slice()?).map_err(map_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Decode one on-wire frame back to PCM. `bits` must be exactly
    /// [`fec_frame_bytes`] bytes long.
    fn decode_bits<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let pcm = self.inner.decode_bits(bits).map_err(map_err)?;
        Ok(pcm.into_pyarray_bound(py))
    }

    /// Clear cross-frame state. Call between independent streams /
    /// callers sharing one vocoder instance.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("Vocoder(rate=Rate.{:?})", self.inner.rate())
    }
}

/// Chunk-driven streaming encoder. Buffers caller-supplied PCM into
/// 160-sample frames and emits one wire-byte blob per completed
/// frame. Pass any chunk size you have; partial frames are held until
/// the next [`push`].
#[pyclass(name = "LiveEncoder", unsendable)]
pub struct PyLiveEncoder {
    inner: bv::LiveEncoder,
}

#[pymethods]
impl PyLiveEncoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::LiveEncoder::new(rate.into()) }
    }

    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    #[getter]
    fn pending_samples(&self) -> usize {
        self.inner.pending_samples()
    }

    /// Push PCM. Returns a list of `bytes`, one per completed frame.
    /// Frames that fail to encode raise `ValueError` at the failing
    /// frame's slot.
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        pcm: PyReadonlyArray1<'py, i16>,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let mut out = Vec::new();
        for r in self.inner.push(pcm.as_slice()?) {
            let bytes = r.map_err(map_err)?;
            out.push(PyBytes::new_bound(py, &bytes));
        }
        Ok(out)
    }

    /// Pad the residue with zeros and emit any pending frame.
    /// Returns the frame bytes or `None` if the residue was empty.
    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        Ok(self
            .inner
            .flush()
            .map_err(map_err)?
            .map(|b| PyBytes::new_bound(py, &b)))
    }

    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Chunk-driven streaming decoder. Buffers caller-supplied wire
/// bytes into [`fec_frame_bytes`]-sized frames and emits PCM blocks
/// per decoded frame.
#[pyclass(name = "LiveDecoder", unsendable)]
pub struct PyLiveDecoder {
    inner: bv::LiveDecoder,
}

#[pymethods]
impl PyLiveDecoder {
    #[new]
    fn new(rate: PyRate) -> Self {
        Self { inner: bv::LiveDecoder::new(rate.into()) }
    }

    #[getter]
    fn rate(&self) -> PyRate {
        self.inner.rate().into()
    }

    #[getter]
    fn pending_bytes(&self) -> usize {
        self.inner.pending_bytes()
    }

    /// Push wire bytes. Returns a list of `np.int16` arrays, one per
    /// completed frame.
    fn push<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Vec<Bound<'py, PyArray1<i16>>>> {
        let mut out = Vec::new();
        for r in self.inner.push(bits) {
            let pcm = r.map_err(map_err)?;
            out.push(pcm.into_pyarray_bound(py));
        }
        Ok(out)
    }

    fn discard_pending(&mut self) {
        self.inner.discard_pending();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Wire-format bridge — decodes one rate and re-encodes to another.
/// Used for Phase 1 IMBE ⇄ Phase 2 AMBE+2 conversion and for same-
/// codec FEC ↔ no-FEC pairs.
#[pyclass(name = "Transcoder", unsendable)]
pub struct PyTranscoder {
    inner: bv::Transcoder,
}

#[pymethods]
impl PyTranscoder {
    #[new]
    fn new(from: PyRate, to: PyRate) -> PyResult<Self> {
        let inner = bv::Transcoder::new(from.into(), to.into()).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Transcode one input frame to the destination rate.
    fn transcode<'py>(
        &mut self,
        py: Python<'py>,
        bits: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let out = self.inner.transcode(bits).map_err(map_err)?;
        Ok(PyBytes::new_bound(py, &out))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pymodule]
fn _blip25_mbe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRate>()?;
    m.add_class::<PyVocoder>()?;
    m.add_class::<PyLiveEncoder>()?;
    m.add_class::<PyLiveDecoder>()?;
    m.add_class::<PyTranscoder>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
