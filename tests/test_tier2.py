"""Coverage for Tier-2 surface: setters, enums, and diagnostics."""

import numpy as np
import pytest

import blip25_mbe


def test_default_setters_match_library_defaults() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.IMBE_7200X4400)
    # `Vocoder::new` enables enhancement (Classical) and spectral
    # subtraction by default; everything else defaults off.
    assert vc.enhancement == blip25_mbe.EnhancementMode.CLASSICAL
    assert vc.spectral_subtraction is True
    assert vc.tone_detection is False
    assert vc.silence_dispatch is False
    assert vc.chip_compat is False
    assert vc.pyin_pitch is False
    assert vc.ambe_plus2_synth == blip25_mbe.AmbePlus2Synth.AMBE_PLUS
    assert vc.repeat_reset_after is None


def test_setters_round_trip() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)

    vc.set_enhancement(blip25_mbe.EnhancementMode.NONE)
    assert vc.enhancement == blip25_mbe.EnhancementMode.NONE

    vc.set_spectral_subtraction(False)
    assert vc.spectral_subtraction is False

    vc.set_tone_detection(True)
    assert vc.tone_detection is True

    vc.set_amp_ema_alpha(0.5)
    assert vc.amp_ema_alpha == pytest.approx(0.5)

    vc.set_repeat_reset_after(50)
    assert vc.repeat_reset_after == 50
    vc.set_repeat_reset_after(None)
    assert vc.repeat_reset_after is None

    vc.set_ambe_plus2_synth(blip25_mbe.AmbePlus2Synth.BASELINE)
    assert vc.ambe_plus2_synth == blip25_mbe.AmbePlus2Synth.BASELINE


def test_last_output_kind_after_encode() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.IMBE_7200X4400)
    # No encode yet → diagnostic is None.
    assert vc.last_output_kind() is None

    silent = np.zeros(vc.frame_samples, dtype=np.int16)
    vc.encode_pcm(silent)
    kind = vc.last_output_kind()
    assert kind in ("voice", "silence", "tone")


def test_tone_detection_surfaces_id_amplitude() -> None:
    # Half-rate AMBE+2 + tone detection ON: a clean Annex T sine
    # should classify as a tone and populate `last_tone_detection`.
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)
    vc.set_tone_detection(True)

    # Generate ~1031 Hz tone — the Motorola standard test pattern.
    n = vc.frame_samples
    t = np.arange(n) / 8000.0
    pcm = (10000 * np.sin(2 * np.pi * 1031.25 * t)).astype(np.int16)

    vc.encode_pcm(pcm)
    detection = vc.last_tone_detection()
    assert detection is not None
    tone_id, amp = detection
    assert 0 <= tone_id <= 255
    assert 0 <= amp <= 127
