import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RemoteAccessBoundaryTests(unittest.TestCase):
    def test_container_enables_xsrf_and_run_scripts_bind_loopback_by_default(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        run_script = (ROOT / "build-and-run.sh").read_text()
        readme = (ROOT / "README.md").read_text()

        self.assertIn("--server.enableXsrfProtection=true", dockerfile)
        self.assertNotIn('docker run -d -p "${PORT}:8501"', run_script)
        self.assertIn('127.0.0.1:${PORT}:8501', run_script)
        self.assertIn("no user authentication or authorization", readme)
        self.assertIn("authenticating, TLS-terminating reverse proxy", readme)


if __name__ == "__main__":
    unittest.main()
