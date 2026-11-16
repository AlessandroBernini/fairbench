"""Tests for fairbench.docker — Docker container support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestDockerDetection:
    """Test Docker detection functions."""

    def test_detect_docker_available_true(self):
        """Test detecting Docker when it's installed."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Docker version 24.0"

        with patch("subprocess.run", return_value=mock_result):
            from fairbench.docker import detect_docker_available

            assert detect_docker_available() is True

    def test_detect_docker_available_false(self):
        """Test detecting Docker when it's not installed."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            from fairbench.docker import detect_docker_available

            assert detect_docker_available() is False

    def test_detect_docker_file_not_found(self):
        """Test detecting Docker when command not found."""
        import subprocess  # noqa: F401 (used in side_effect)

        with patch(
            "subprocess.run", side_effect=FileNotFoundError("docker not found")
        ):
            from fairbench.docker import detect_docker_available

            assert detect_docker_available() is False


class TestGpuSupportDetection:
    """Test GPU support detection."""

    def test_detect_nvidia_gpu(self):
        """Test detecting NVIDIA GPU support."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result), patch(
            "platform.system", return_value="Linux"
        ):
            from fairbench.docker import detect_gpu_support

            assert detect_gpu_support() == "nvidia"

    def test_detect_rocm_gpu(self):
        """Test detecting AMD ROCm GPU support."""
        mock_result = MagicMock()
        mock_result.returncode = 1  # nvidia-smi not found

        with patch("subprocess.run", return_value=mock_result), patch(
            "platform.system", return_value="Linux"
        ), patch("os.path.exists", return_value=True):
            from fairbench.docker import detect_gpu_support

            assert detect_gpu_support() == "rocm"

    def test_detect_none_gpu(self):
        """Test detecting no GPU support."""
        mock_result = MagicMock()
        mock_result.returncode = 1  # nvidia-smi not found

        with patch("subprocess.run", return_value=mock_result), patch(
            "platform.system", return_value="Linux"
        ), patch("os.path.exists", return_value=False):
            from fairbench.docker import detect_gpu_support

            assert detect_gpu_support() == "none"


class TestMacOsDetection:
    """Test macOS Docker limitation detection."""

    def test_detect_macos_docker_limitation(self):
        """Test detecting Docker on macOS (no GPU)."""
        with patch("platform.system", return_value="Darwin"):
            from fairbench.docker import detect_macos_docker_limitation

            assert detect_macos_docker_limitation() is True

    def test_detect_linux_no_limitation(self):
        """Test Linux doesn't have macOS limitation."""
        with patch("platform.system", return_value="Linux"):
            from fairbench.docker import detect_macos_docker_limitation

            assert detect_macos_docker_limitation() is False


class TestVerifyDockerSetup:
    """Test Docker setup verification."""

    def test_verify_docker_not_available(self):
        """Test verification when Docker not available."""
        with patch(
            "fairbench.docker.detect_docker_available", return_value=False
        ):
            from fairbench.docker import verify_docker_setup

            info = verify_docker_setup()
            assert info["docker_available"] == "No"

    def test_verify_docker_with_gpu(self):
        """Test verification with GPU support."""
        with patch(
            "fairbench.docker.detect_docker_available", return_value=True
        ), patch("fairbench.docker.detect_gpu_support", return_value="nvidia"), patch(
            "fairbench.docker.detect_macos_docker_limitation", return_value=False
        ):
            from fairbench.docker import verify_docker_setup

            info = verify_docker_setup()
            assert info["docker_available"] == "Yes"
            assert info["gpu_support"] == "Nvidia"

    def test_verify_macos_gpu_limitation(self):
        """Test macOS GPU limitation note."""
        with patch(
            "fairbench.docker.detect_docker_available", return_value=True
        ), patch("fairbench.docker.detect_gpu_support", return_value="none"), patch(
            "fairbench.docker.detect_macos_docker_limitation", return_value=True
        ):
            from fairbench.docker import verify_docker_setup

            info = verify_docker_setup()
            assert "note" in info


class TestBuildDockerImage:
    """Test Docker image building."""

    def test_build_success(self, tmp_path):
        """Test successful Docker image build."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.12-slim\n")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch(
            "fairbench.docker.Path.exists", return_value=True
        ), patch.object(
            __import__("pathlib", fromlist=["Path"]).Path,
            "__truediv__",
            return_value=dockerfile,
        ), patch("subprocess.run", return_value=mock_result):
            from fairbench.docker import build_docker_image

            result = build_docker_image("test-image")
            assert result is True

    def test_build_missing_dockerfile(self):
        """Test build fails when Dockerfile not found."""
        with patch(
            "fairbench.docker.Path.exists", return_value=False
        ):
            from fairbench.docker import build_docker_image

            result = build_docker_image("test-image")
            assert result is False


class TestRunInDocker:
    """Test Docker container execution."""

    def test_run_cpu_only(self):
        """Test running in CPU-only mode."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            from fairbench.docker import run_in_docker

            exit_code = run_in_docker(
                image_name="test-image",
                model_path="/tmp/models/model.gguf",
                category="quality",
                lite_mode="disabled",
                gpu_support="none",
                ci_mode=False,
            )
            assert exit_code == 0

    def test_run_with_gpu(self):
        """Test running with GPU support."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            from fairbench.docker import run_in_docker

            exit_code = run_in_docker(
                image_name="test-image",
                model_path="/tmp/models/model.gguf",
                category="quality",
                lite_mode="enabled",
                gpu_support="nvidia",
                ci_mode=True,
            )
            assert exit_code == 0

    def test_run_timeout(self):
        """Test handling Docker timeout."""
        import subprocess

        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 3600)
        ):
            from fairbench.docker import run_in_docker

            exit_code = run_in_docker(
                image_name="test-image",
                model_path="/tmp/models/model.gguf",
                category="quality",
                lite_mode="disabled",
                gpu_support="none",
                ci_mode=False,
            )
            assert exit_code == 1
