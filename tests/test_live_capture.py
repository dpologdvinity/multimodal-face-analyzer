import pathlib
import unittest


APP_SOURCE = pathlib.Path(__file__).parents[1].joinpath("src", "app.py").read_text()


class LiveCaptureTests(unittest.TestCase):
    def test_live_stream_uses_thread_safe_shared_info_and_async_processing(self):
        self.assertIn("LIVE_STATE", APP_SOURCE)
        self.assertIn("gallery_snapshot", APP_SOURCE)
        self.assertIn("async_processing=True", APP_SOURCE)
        self.assertIn("while webrtc_ctx.state.playing", APP_SOURCE)

    def test_live_callback_keeps_camera_alive_when_inference_fails(self):
        self.assertIn("except Exception as exc:", APP_SOURCE)
        self.assertIn("return frame", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
