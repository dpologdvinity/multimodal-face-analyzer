import unittest

import numpy as np

from src.inference import (
    AUDIO_AROUSAL_LOUD_RMS,
    AUDIO_AROUSAL_QUIET_RMS,
    VoiceFaceFusion,
    audio_frame_to_mono_float,
    classify_voice_arousal,
    fuse_voice_and_emotion,
)


class VoiceFusionTests(unittest.TestCase):
    def test_audio_frame_to_mono_float_normalizes_integer_stereo(self):
        samples = np.array([[32767, -32768], [32767, -32768]], dtype=np.int16)

        result = audio_frame_to_mono_float(samples)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(float(result[0]), 32767 / 32767, places=5)
        self.assertAlmostEqual(float(result[1]), -32768 / 32767, places=5)

    def test_classify_voice_arousal_uses_quiet_and_loud_thresholds(self):
        self.assertEqual(classify_voice_arousal(AUDIO_AROUSAL_QUIET_RMS - 0.001), "QUIET")
        self.assertEqual(classify_voice_arousal(AUDIO_AROUSAL_QUIET_RMS), "SPEAKING")
        self.assertEqual(classify_voice_arousal(AUDIO_AROUSAL_LOUD_RMS), "LOUD")

    def test_fusion_compares_voice_arousal_with_face_emotion_arousal(self):
        self.assertEqual(fuse_voice_and_emotion("QUIET", "neutral"), "consistent")
        self.assertEqual(fuse_voice_and_emotion("LOUD", "neutral"), "inconsistent")
        self.assertEqual(fuse_voice_and_emotion("LOUD", "happy"), "consistent")
        self.assertIsNone(fuse_voice_and_emotion("LOUD", "unknown"))

    def test_voice_buffer_discards_audio_older_than_configured_window(self):
        fusion = VoiceFaceFusion(window_seconds=0.5)
        fusion.ingest_audio(np.ones(4, dtype=np.float32), sample_rate=4)
        fusion.ingest_audio(np.zeros(4, dtype=np.float32), sample_rate=4)

        self.assertEqual(fusion.current_arousal(), "QUIET")

        fusion.reset()
        self.assertEqual(fusion.current_arousal(), "QUIET")


if __name__ == "__main__":
    unittest.main()
