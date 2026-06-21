//! Python bindings for `blip25-mbe`.
//!
//! Exposes the chip-shaped façade — `Rate`, `Vocoder`, `Transcoder`,
//! `LiveEncoder`, `LiveDecoder` — with zero-copy numpy interop on the
//! PCM boundary. Tier-1 surface only; setters, the builder, and
//! parameter-domain entry points will land in a follow-on.

use blip25_mbe::codecs::mbe_baseline::analysis::DenoiseKind as BDenoiseKind;
use blip25_mbe::dvsi_soft_decision as sd;
use blip25_mbe::enhancement::{ClassicalConfig, EnhancementMode};
use blip25_mbe::vocoder::{self as bv, AmbePlus2Synth as BSynth, AnalysisOutputKind, Rate as BRate};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

/// Map a `VocoderError` to a Python `ValueError` with the upstream
/// `Display` text — preserves the actionable message without
/// inventing a new exception hierarchy.
fn map_err(e: bv::VocoderError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Map a `SoftDecisionError` to a Python `ValueError`, same policy as
/// [`map_err`].
fn map_sd_err(e: sd::SoftDecisionError) -> PyErr {
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
    /// AMBE+2 info-only (7-byte frame, 2 450 bps). The 49 info bits are
    /// packed in DVSI **r34 column-interleave** order — byte-exact with
    /// DVSI's chip rate-index 34 no-FEC stream, *not* naive MSB-first
    /// sequential. Consumers needing natural / "AMBE_d" order (mbelib,
    /// IDAS/NXDN over-the-air) must de-interleave first.
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
        // `Rate` is `#[non_exhaustive]` upstream; the catch-all
        // covers future variants when this crate is rebuilt against
        // a newer `blip25-mbe`. Currently unreachable.
        #[allow(unreachable_patterns)]
        match r {
            BRate::Imbe7200x4400 => PyRate::Imbe7200x4400,
            BRate::Imbe4400x4400 => PyRate::Imbe4400x4400,
            BRate::AmbePlus2_3600x2450 => PyRate::AmbePlus2_3600x2450,
            BRate::AmbePlus2_2450x2450 => PyRate::AmbePlus2_2450x2450,
            other => panic!("blip25-mbe-py: unmapped upstream Rate variant {other:?}"),
        }
    }
}

/// AMBE+2 unvoiced/voiced synthesis variant. Default is
/// `AmbePlus` — modern AMBE+ / AMBE+2 sound with US5701390 phase
/// regen. `Baseline` matches mbelib's half-rate behavior (no
/// Hilbert phase regen).
#[pyclass(eq, eq_int, frozen, name = "AmbePlus2Synth")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyAmbePlus2Synth {
    #[pyo3(name = "AMBE_PLUS")]
    AmbePlus,
    #[pyo3(name = "BASELINE")]
    Baseline,
}

impl From<PyAmbePlus2Synth> for BSynth {
    fn from(s: PyAmbePlus2Synth) -> Self {
        match s {
            PyAmbePlus2Synth::AmbePlus => BSynth::AmbePlus,
            PyAmbePlus2Synth::Baseline => BSynth::Baseline,
        }
    }
}

impl From<BSynth> for PyAmbePlus2Synth {
    fn from(s: BSynth) -> Self {
        #[allow(unreachable_patterns)]
        match s {
            BSynth::AmbePlus => PyAmbePlus2Synth::AmbePlus,
            BSynth::Baseline => PyAmbePlus2Synth::Baseline,
            other => panic!("blip25-mbe-py: unmapped upstream AmbePlus2Synth variant {other:?}"),
        }
    }
}

/// Pre-analysis denoiser gain rule (blip25-mbe 0.2.0). `LOG_MMSE` is
/// the default (least musical noise); `WIENER` and `SPEC_SUB` are kept
/// for A/B against the §3.4 baseline. Passing any of these to
/// [`Vocoder.set_denoise_kind`] enables the (opt-in) denoiser front-end.
#[pyclass(eq, eq_int, frozen, name = "DenoiseKind")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyDenoiseKind {
    #[pyo3(name = "LOG_MMSE")]
    LogMmse,
    #[pyo3(name = "WIENER")]
    Wiener,
    #[pyo3(name = "SPEC_SUB")]
    SpecSub,
}

impl From<PyDenoiseKind> for BDenoiseKind {
    fn from(k: PyDenoiseKind) -> Self {
        match k {
            PyDenoiseKind::LogMmse => BDenoiseKind::LogMmse,
            PyDenoiseKind::Wiener => BDenoiseKind::Wiener,
            PyDenoiseKind::SpecSub => BDenoiseKind::SpecSub,
        }
    }
}

/// Post-decoder PCM enhancement mode. `NONE` is spec-faithful (no
/// post-processing); `CLASSICAL` applies the default biquad +
/// peaking-EQ + output-gain chain (no compressor, fade enabled).
/// `CLASSICAL` is the upstream library default.
///
/// Fine-grained tuning of the classical chain (custom biquads,
/// compressor knobs, fade samples) isn't exposed from Python yet —
/// open an issue if you need it.
#[pyclass(eq, eq_int, frozen, name = "EnhancementMode")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEnhancementMode {
    #[pyo3(name = "NONE")]
    NoneMode,
    #[pyo3(name = "CLASSICAL")]
    Classical,
}

impl From<PyEnhancementMode> for EnhancementMode {
    fn from(m: PyEnhancementMode) -> Self {
        match m {
            PyEnhancementMode::NoneMode => EnhancementMode::None,
            PyEnhancementMode::Classical => EnhancementMode::Classical(ClassicalConfig::default()),
        }
    }
}

impl From<&EnhancementMode> for PyEnhancementMode {
    fn from(m: &EnhancementMode) -> Self {
        match m {
            EnhancementMode::None => PyEnhancementMode::NoneMode,
            EnhancementMode::Classical(_) => PyEnhancementMode::Classical,
        }
    }
}

fn output_kind_str(k: AnalysisOutputKind) -> &'static str {
    // `AnalysisOutputKind` is `#[non_exhaustive]` upstream; the
    // catch-all guards against future variants when this crate is
    // rebuilt against a newer `blip25-mbe`.
    #[allow(unreachable_patterns)]
    match k {
        AnalysisOutputKind::Voice => "voice",
        AnalysisOutputKind::Silence => "silence",
        AnalysisOutputKind::Tone { .. } => "tone",
        _ => "unknown",
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

    // ── encoder knobs ───────────────────────────────────────────

    #[getter]
    fn tone_detection(&self) -> bool { self.inner.tone_detection() }

    fn set_tone_detection(&mut self, on: bool) { self.inner.set_tone_detection(on); }

    #[getter]
    fn spectral_subtraction(&self) -> bool { self.inner.spectral_subtraction() }

    fn set_spectral_subtraction(&mut self, on: bool) { self.inner.set_spectral_subtraction(on); }

    #[getter]
    fn silence_dispatch(&self) -> bool { self.inner.silence_dispatch() }

    fn set_silence_dispatch(&mut self, on: bool) { self.inner.set_silence_dispatch(on); }

    #[getter]
    fn chip_compat(&self) -> bool { self.inner.chip_compat() }

    fn set_chip_compat(&mut self, on: bool) { self.inner.set_chip_compat(on); }

    #[getter]
    fn chip_compat_spectral_clamp(&self) -> bool { self.inner.chip_compat_spectral_clamp() }

    fn set_chip_compat_spectral_clamp(&mut self, on: bool) {
        self.inner.set_chip_compat_spectral_clamp(on);
    }

    #[getter]
    fn pitch_silence_override(&self) -> bool { self.inner.pitch_silence_override() }

    fn set_pitch_silence_override(&mut self, on: bool) {
        self.inner.set_pitch_silence_override(on);
    }

    #[getter]
    fn default_pitch_on_silence(&self) -> bool { self.inner.default_pitch_on_silence() }

    fn set_default_pitch_on_silence(&mut self, on: bool) {
        self.inner.set_default_pitch_on_silence(on);
    }

    #[getter]
    fn pyin_pitch(&self) -> bool { self.inner.pyin_pitch() }

    fn set_pyin_pitch(&mut self, on: bool) { self.inner.set_pyin_pitch(on); }

    #[getter]
    fn amp_ema_alpha(&self) -> f64 { self.inner.amp_ema_alpha() }

    fn set_amp_ema_alpha(&mut self, alpha: f64) { self.inner.set_amp_ema_alpha(alpha); }

    #[getter]
    fn repeat_reset_after(&self) -> Option<u32> { self.inner.repeat_reset_after() }

    #[pyo3(signature = (n=None))]
    fn set_repeat_reset_after(&mut self, n: Option<u32>) {
        self.inner.set_repeat_reset_after(n);
    }

    #[getter]
    fn ambe_plus2_synth(&self) -> PyAmbePlus2Synth { self.inner.ambe_plus2_synth().into() }

    fn set_ambe_plus2_synth(&mut self, gen: PyAmbePlus2Synth) {
        self.inner.set_ambe_plus2_synth(gen.into());
    }

    #[getter]
    fn enhancement(&self) -> PyEnhancementMode { self.inner.enhancement().into() }

    fn set_enhancement(&mut self, mode: PyEnhancementMode) {
        self.inner.set_enhancement(mode.into());
    }

    // ── encode-quality stack (blip25-mbe 0.2.0) ─────────────────
    // Defaulted ON in `Vocoder::new()` for AMBE+2 production; each is
    // individually opt-out here. Full-rate IMBE stays spec-faithful.

    #[getter]
    fn pitch_decide_escape(&self) -> bool { self.inner.pitch_decide_escape() }

    fn set_pitch_decide_escape(&mut self, on: bool) {
        self.inner.set_pitch_decide_escape(on);
    }

    #[getter]
    fn pitch_subsample(&self) -> bool { self.inner.pitch_subsample() }

    fn set_pitch_subsample(&mut self, on: bool) { self.inner.set_pitch_subsample(on); }

    /// `True` runs the §0.4 `E_R` pitch refinement (spec); `False`
    /// emits the raw §0.3 estimate (the production stack default).
    #[getter]
    fn pitch_refine(&self) -> bool { self.inner.pitch_refine_enabled() }

    fn set_pitch_refine(&mut self, on: bool) { self.inner.set_pitch_refine(on); }

    /// Hard-bounded M(ξ) loudness-graded Θ relaxation (default ON in
    /// the stack; cannot mute).
    #[getter]
    fn vuv_mxi_grade(&self) -> bool { self.inner.vuv_mxi_grade_enabled() }

    fn set_vuv_mxi_grade(&mut self, on: bool) { self.inner.set_vuv_mxi_grade(on); }

    /// Eq. 37 V/UV pitch/band Θ rolloff coefficient (spec default
    /// 0.3096; 0.0 = chip-observed no-rolloff). Write-only upstream.
    fn set_vuv_pitch_coef(&mut self, c: f64) { self.inner.set_vuv_pitch_coef(c); }

    /// Fractional §0.5 band-edge coverage weighting (default OFF;
    /// opt-in loudness/shape lever). Write-only upstream.
    fn set_amp_frac_band_edges(&mut self, on: bool) {
        self.inner.set_amp_frac_band_edges(on);
    }

    /// Flat +0.9 dB chip-measured level normalization (default OFF;
    /// AMBE+2 only). Write-only upstream.
    fn set_level_scale(&mut self, on: bool) { self.inner.set_level_scale(on); }

    /// Silence shape-zeroing on silent analysis windows (default OFF).
    /// Write-only upstream.
    fn set_silence_shape_zero(&mut self, on: bool) {
        self.inner.set_silence_shape_zero(on);
    }

    // ── denoiser front-ends (blip25-mbe 0.2.0, all opt-in) ──────
    // General-DSP front-ends on the input PCM, ahead of the codec;
    // transparent on clean speech, *exceed* levers on noisy field audio.

    #[getter]
    fn denoise(&self) -> bool { self.inner.denoise() }

    /// Enable/disable the §3.4 pre-analysis log-MMSE denoiser.
    fn set_denoise(&mut self, on: bool) { self.inner.set_denoise(on); }

    /// Select the denoiser gain rule (see [`DenoiseKind`]) and enable
    /// it. For A/B sweeps.
    fn set_denoise_kind(&mut self, kind: PyDenoiseKind) {
        self.inner.set_denoise_kind(kind.into());
    }

    #[getter]
    fn hum_notch(&self) -> bool { self.inner.hum_notch() }

    /// Enable/disable the 60/120 Hz (US) mains-hum notch. Use
    /// [`set_hum_notch_mains`] for 50/100 Hz (EU).
    fn set_hum_notch(&mut self, on: bool) { self.inner.set_hum_notch(on); }

    /// Enable the hum notch at a specific mains fundamental (e.g. 50.0
    /// for EU); also nulls `2·mains_hz`.
    fn set_hum_notch_mains(&mut self, mains_hz: f64) {
        self.inner.set_hum_notch_mains(mains_hz);
    }

    // ── diagnostics ────────────────────────────────────────────

    /// `"voice"` / `"silence"` / `"tone"` for the last encoded frame,
    /// or `None` if no encode has run since the last [`reset`].
    fn last_output_kind(&self) -> Option<&'static str> {
        self.inner
            .last_stats()
            .analysis
            .as_ref()
            .map(|a| output_kind_str(a.output))
    }

    /// `(id, amplitude)` for the last frame's Annex T tone detection,
    /// or `None` if tone detection is disabled or no tone was found.
    fn last_tone_detection(&self) -> Option<(u8, u8)> {
        self.inner
            .last_stats()
            .analysis
            .as_ref()
            .and_then(|a| a.tone_detect)
            .map(|t| (t.id, t.amplitude))
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

// ── DVSI soft-decision chip handoff (blip25-mbe 0.2.0) ──────────────
// The 4-bit soft-decision (LLR) packet format for soft-FEC interchange
// with DVSI AMBE-2000/2020/3000 hardware. Exposed as the
// `blip25_mbe.dvsi_soft_decision` submodule.

/// The 12 overhead words of an AMBE-2000/2020 soft-decision packet,
/// minus the fixed `0x13EC` header. The five `rate_info` words and the
/// control words are chip-/rate-specific and caller-supplied.
#[pyclass(name = "SdPacketHeader")]
#[derive(Clone)]
pub struct PySdPacketHeader {
    inner: sd::SdPacketHeader,
}

#[pymethods]
impl PySdPacketHeader {
    /// Construct a header. The common case passes only `rate_info`
    /// (the five rate-control words); all other overhead fields default
    /// to zero. `rate_info` must be a sequence of exactly five `u16`.
    #[new]
    #[pyo3(signature = (
        rate_info = [0u16; 5],
        *,
        power_control = 0,
        control_word1 = 0,
        dtmf_control = 0,
        control_word2 = 0,
    ))]
    fn new(
        rate_info: [u16; 5],
        power_control: u8,
        control_word1: u8,
        dtmf_control: u16,
        control_word2: u16,
    ) -> Self {
        Self {
            inner: sd::SdPacketHeader {
                power_control,
                control_word1,
                rate_info,
                dtmf_control,
                control_word2,
            },
        }
    }

    #[getter]
    fn power_control(&self) -> u8 { self.inner.power_control }
    #[setter]
    fn set_power_control(&mut self, v: u8) { self.inner.power_control = v; }

    #[getter]
    fn control_word1(&self) -> u8 { self.inner.control_word1 }
    #[setter]
    fn set_control_word1(&mut self, v: u8) { self.inner.control_word1 = v; }

    #[getter]
    fn rate_info(&self) -> Vec<u16> { self.inner.rate_info.to_vec() }
    #[setter]
    fn set_rate_info(&mut self, v: [u16; 5]) { self.inner.rate_info = v; }

    #[getter]
    fn dtmf_control(&self) -> u16 { self.inner.dtmf_control }
    #[setter]
    fn set_dtmf_control(&mut self, v: u16) { self.inner.dtmf_control = v; }

    #[getter]
    fn control_word2(&self) -> u16 { self.inner.control_word2 }
    #[setter]
    fn set_control_word2(&mut self, v: u16) { self.inner.control_word2 = v; }

    fn __eq__(&self, other: &Self) -> bool { self.inner == other.inner }

    fn __repr__(&self) -> String {
        format!(
            "SdPacketHeader(rate_info={:?}, power_control={}, control_word1={}, dtmf_control={}, control_word2={})",
            self.inner.rate_info,
            self.inner.power_control,
            self.inner.control_word1,
            self.inner.dtmf_control,
            self.inner.control_word2,
        )
    }
}

/// Convert one native soft bit (`i8` LLR: sign = hard decision with
/// `> 0` meaning `1`, magnitude = confidence) into a DVSI 4-bit
/// soft-decision value in `0..=15` (offset-binary, MSB = hard bit).
#[pyfunction]
fn llr_to_sd_nibble(llr: i8) -> u8 { sd::llr_to_sd_nibble(llr) }

/// Inverse of [`llr_to_sd_nibble`]: map a DVSI 4-bit SD value back to a
/// representative `i8` LLR. Only the low nibble of `n` is used.
#[pyfunction]
fn sd_nibble_to_llr(n: u8) -> i8 { sd::sd_nibble_to_llr(n) }

/// Pack soft channel bits (one `int8` LLR per bit, `SD0` first) into a
/// 60-word DVSI soft-decision decoder packet (returned as a `uint16`
/// array). Fewer than 192 bits is normal; unused slots fill with
/// most-confident-0. Raises `ValueError` if more than 192 bits given.
#[pyfunction]
fn pack_channel_bits<'py>(
    py: Python<'py>,
    channel_llrs: PyReadonlyArray1<'py, i8>,
    header: &PySdPacketHeader,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let words =
        sd::pack_channel_bits(channel_llrs.as_slice()?, &header.inner).map_err(map_sd_err)?;
    Ok(words.to_vec().into_pyarray_bound(py))
}

/// Unpack a 60-word DVSI soft-decision packet (`uint16` array) into its
/// `(SdPacketHeader, int8[192] LLRs)`. Verifies the `0x13EC` header.
#[pyfunction]
fn unpack_packet<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'py, u16>,
) -> PyResult<(PySdPacketHeader, Bound<'py, PyArray1<i8>>)> {
    let slice = words.as_slice()?;
    let arr: &[u16; sd::SD_PACKET_WORDS] = slice.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected {} packet words, got {}",
            sd::SD_PACKET_WORDS,
            slice.len()
        ))
    })?;
    let (header, llrs) = sd::unpack_packet(arr).map_err(map_sd_err)?;
    Ok((PySdPacketHeader { inner: header }, llrs.to_vec().into_pyarray_bound(py)))
}

/// Serialize a 60-word packet (`uint16` array) to 120 big-endian bytes
/// (high byte of each word first — the host wire order).
#[pyfunction]
fn packet_to_bytes<'py>(
    py: Python<'py>,
    words: PyReadonlyArray1<'py, u16>,
) -> PyResult<Bound<'py, PyBytes>> {
    let slice = words.as_slice()?;
    let arr: &[u16; sd::SD_PACKET_WORDS] = slice.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "expected {} packet words, got {}",
            sd::SD_PACKET_WORDS,
            slice.len()
        ))
    })?;
    Ok(PyBytes::new_bound(py, &sd::packet_to_bytes(arr)))
}

/// Pack soft channel bits (`int8` LLRs) into the raw USB-3000 SD nibble
/// stream (`*_sd.bit` format): two 4-bit values per byte, `SD0` in the
/// high nibble. No header or rate words.
#[pyfunction]
fn pack_nibble_stream<'py>(
    py: Python<'py>,
    channel_llrs: PyReadonlyArray1<'py, i8>,
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new_bound(
        py,
        &sd::pack_nibble_stream(channel_llrs.as_slice()?),
    ))
}

/// Unpack a raw USB-3000 SD nibble stream (`bytes`) into `int8` LLRs,
/// `SD0` first. Inverse of [`pack_nibble_stream`].
#[pyfunction]
fn unpack_nibble_stream<'py>(py: Python<'py>, bytes: &[u8]) -> Bound<'py, PyArray1<i8>> {
    sd::unpack_nibble_stream(bytes).into_pyarray_bound(py)
}

/// Build and register the `dvsi_soft_decision` submodule.
fn register_dvsi_sd(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new_bound(py, "dvsi_soft_decision")?;
    m.add_class::<PySdPacketHeader>()?;
    m.add_function(wrap_pyfunction!(llr_to_sd_nibble, &m)?)?;
    m.add_function(wrap_pyfunction!(sd_nibble_to_llr, &m)?)?;
    m.add_function(wrap_pyfunction!(pack_channel_bits, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_packet, &m)?)?;
    m.add_function(wrap_pyfunction!(packet_to_bytes, &m)?)?;
    m.add_function(wrap_pyfunction!(pack_nibble_stream, &m)?)?;
    m.add_function(wrap_pyfunction!(unpack_nibble_stream, &m)?)?;
    m.add("SD_HEADER", sd::SD_HEADER)?;
    m.add("SD_PACKET_WORDS", sd::SD_PACKET_WORDS)?;
    m.add("SD_OVERHEAD_WORDS", sd::SD_OVERHEAD_WORDS)?;
    m.add("SD_DATA_WORDS", sd::SD_DATA_WORDS)?;
    m.add("SD_SLOTS", sd::SD_SLOTS)?;
    m.add("RATE_33_CHANNEL_BITS", sd::RATE_33_CHANNEL_BITS)?;
    m.add("IMBE_FULL_RATE_CHANNEL_BITS", sd::IMBE_FULL_RATE_CHANNEL_BITS)?;
    m.add("DVSI_P25_FULLRATE_FEC", sd::DVSI_P25_FULLRATE_FEC.to_vec())?;
    m.add("DVSI_P25_FULLRATE_NOFEC", sd::DVSI_P25_FULLRATE_NOFEC.to_vec())?;
    parent.add_submodule(&m)?;
    // Register in sys.modules so `import blip25_mbe._blip25_mbe.dvsi_soft_decision`
    // resolves (pyo3 submodules are not auto-registered).
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("blip25_mbe._blip25_mbe.dvsi_soft_decision", &m)?;
    Ok(())
}

#[pymodule]
fn _blip25_mbe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRate>()?;
    m.add_class::<PyAmbePlus2Synth>()?;
    m.add_class::<PyEnhancementMode>()?;
    m.add_class::<PyDenoiseKind>()?;
    m.add_class::<PyVocoder>()?;
    m.add_class::<PyLiveEncoder>()?;
    m.add_class::<PyLiveDecoder>()?;
    m.add_class::<PyTranscoder>()?;
    register_dvsi_sd(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
