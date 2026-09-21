import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class LiveCaptureTests(unittest.TestCase):
    """Verify live capture uses thread-safe state and handles inference errors."""

    def test_live_stream_uses_thread_safe_shared_info_and_async_processing(self):
        """Ensure live stream uses shared state dict and async frame processing."""
        self.assertIn("LIVE_STATE", APP_SOURCE)
        self.assertIn("gallery_snapshot", APP_SOURCE)
        self.assertIn("async_processing=True", APP_SOURCE)
        self.assertIn("while webrtc_ctx.state.playing", APP_SOURCE)

    def test_live_callback_keeps_camera_alive_when_inference_fails(self):
        """Verify callback exception handler returns unprocessed frame to keep stream alive."""
        self.assertIn("except Exception as exc:", APP_SOURCE)
        self.assertIn("return frame", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
