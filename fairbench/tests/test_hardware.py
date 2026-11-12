"""Tests for hardware detection module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fairbench.hardware import (
    HardwareProfile,
    detect_hardware,
    generate_adaptive_config,
    generate_recommendation,
)


class TestHardwareProfile(unittest.TestCase):
    """Test HardwareProfile dataclass."""

    def test_default_values(self):
        """Test that HardwareProfile has sensible defaults."""
        profile = HardwareProfile()
        self.assertEqual(profile.cpu_model, "Unknown")
        self.assertEqual(profile.cpu_cores, 0)
        self.assertEqual(profile.cpu_threads, 0)
        self.assertEqual(profile.total_ram_gb, 0.0)
        self.assertFalse(profile.gpu_available)
        self.assertIsNone(profile.gpu_type)
        self.assertEqual(profile.gpu_vram_gb, 0.0)
        self.assertEqual(profile.os_platform, "linux")


class TestGenerateAdaptiveConfig(unittest.TestCase):
    """Test adaptive config generation."""

    def test_low_ram_system(self):
        """Test config for low RAM system (< 8 GB)."""
        hw = HardwareProfile(
            cpu_model="Raspberry Pi 4",
            cpu_cores=4,
            cpu_threads=4,
            total_ram_gb=4.0,
            gpu_available=False,
            gpu_type=None,
            gpu_vram_gb=0.0,
            os_platform="linux",
        )
        config = generate_adaptive_config(hw)
        self.assertEqual(config["default_n_ctx"], 4096)  # min
        self.assertEqual(config["default_n_gpu_layers"], 0)
        self.assertEqual(config["default_threads"], 4)  # max(4, 4*0.75=3)
        self.assertEqual(config["default_quant"], "q4_k_m")

    def test_moderate_ram_system(self):
        """Test config for moderate RAM system (8-16 GB)."""
        hw = HardwareProfile(
            cpu_model="Intel i7",
            cpu_cores=8,
            cpu_threads=16,
            total_ram_gb=12.0,
            gpu_available=False,
            gpu_type=None,
            gpu_vram_gb=0.0,
            os_platform="linux",
        )
        config = generate_adaptive_config(hw)
        self.assertEqual(config["default_n_ctx"], 6144)  # 12*512=6144
        self.assertEqual(config["default_n_gpu_layers"], 0)
        self.assertEqual(config["default_threads"], 12)  # 16*0.75=12
        self.assertEqual(config["default_quant"], "q5_k_m")

    def test_high_ram_system(self):
        """Test config for high RAM system (> 16 GB)."""
        hw = HardwareProfile(
            cpu_model="AMD Ryzen 9",
            cpu_cores=16,
            cpu_threads=32,
            total_ram_gb=32.0,
            gpu_available=True,
            gpu_type="nvidia",
            gpu_vram_gb=12.0,
            os_platform="linux",
        )
        config = generate_adaptive_config(hw)
        self.assertEqual(config["default_n_ctx"], 8192)  # max
        self.assertEqual(config["default_n_gpu_layers"], -1)
        self.assertEqual(config["default_threads"], 24)  # 32*0.75=24
        self.assertEqual(config["default_quant"], "q6_k")

    def test_gpu_detection(self):
        """Test GPU offload with NVIDIA GPU."""
        hw = HardwareProfile(
            cpu_model="Intel Xeon",
            cpu_cores=8,
            cpu_threads=16,
            total_ram_gb=16.0,
            gpu_available=True,
            gpu_type="nvidia",
            gpu_vram_gb=8.0,
            os_platform="linux",
        )
        config = generate_adaptive_config(hw)
        self.assertEqual(config["default_n_gpu_layers"], -1)

    def test_context_window_scaling(self):
        """Test context window scales with RAM."""
        hw_low = HardwareProfile(total_ram_gb=4.0)
        hw_high = HardwareProfile(total_ram_gb=32.0)

        config_low = generate_adaptive_config(hw_low)
        config_high = generate_adaptive_config(hw_high)

        self.assertGreater(config_high["default_n_ctx"], config_low["default_n_ctx"])
        self.assertGreaterEqual(config_high["default_n_ctx"], 4096)
        self.assertLessEqual(config_high["default_n_ctx"], 8192)


class TestGenerateRecommendation(unittest.TestCase):
    """Test recommendation generation."""

    def test_model_fits_in_ram(self):
        """Test recommendation when model fits in RAM."""
        hw = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_threads=4,
            total_ram_gb=16.0,
            gpu_available=True,
            gpu_type="apple-silicon",
            gpu_vram_gb=8.0,
            os_platform="darwin",
        )
        recs = generate_recommendation(hw, model_size_gb=4.0)
        self.assertIsNone(recs.get("warning"))
        self.assertEqual(recs["gpu_layers"], -1)
        self.assertEqual(recs["quant"], "q6_k")

    def test_model_exceeds_ram(self):
        """Test warning when model exceeds RAM."""
        hw = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=4,
            cpu_threads=4,
            total_ram_gb=4.0,
            gpu_available=False,
            gpu_type=None,
            gpu_vram_gb=0.0,
            os_platform="linux",
        )
        recs = generate_recommendation(hw, model_size_gb=6.0)
        self.assertIsNotNone(recs.get("warning"))
        self.assertIn("exceeds", recs["warning"])

    def test_gpu_vram_exceeded(self):
        """Test recommendation when model exceeds GPU VRAM."""
        hw = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=8,
            cpu_threads=16,
            total_ram_gb=16.0,
            gpu_available=True,
            gpu_type="nvidia",
            gpu_vram_gb=4.0,
            os_platform="linux",
        )
        recs = generate_recommendation(hw, model_size_gb=8.0)
        self.assertEqual(recs["gpu_layers"], "partial")
        self.assertIn("exceeds", recs["gpu_note"])


class TestDetectHardwareIntegration(unittest.TestCase):
    """Integration tests for hardware detection (mocked)."""

    @patch("fairbench.hardware._detect_cpu")
    @patch("fairbench.hardware._detect_ram")
    @patch("fairbench.hardware._detect_gpu_nvidia")
    @patch("fairbench.hardware._detect_gpu_apple")
    @patch("fairbench.hardware._detect_gpu_amd")
    def test_detect_hardware_all_zeros(
        self,
        mock_amd,
        mock_apple,
        mock_nvidia,
        mock_ram,
        mock_cpu,
    ):
        """Test hardware detection with mocked zero values."""
        mock_cpu.return_value = ("Test CPU", 4, 4)
        mock_ram.return_value = 8.0
        mock_nvidia.return_value = (False, None, 0.0)
        mock_apple.return_value = (False, None, 0.0)
        mock_amd.return_value = (False, None, 0.0)

        hw = detect_hardware()
        self.assertEqual(hw.cpu_model, "Test CPU")
        self.assertEqual(hw.cpu_cores, 4)
        self.assertEqual(hw.total_ram_gb, 8.0)
        self.assertFalse(hw.gpu_available)

    @patch("fairbench.hardware._detect_cpu")
    @patch("fairbench.hardware._detect_ram")
    @patch("fairbench.hardware._detect_gpu_nvidia")
    def test_detect_hardware_with_nvidia(
        self, mock_nvidia, mock_ram, mock_cpu
    ):
        """Test hardware detection with NVIDIA GPU."""
        mock_cpu.return_value = ("Intel i7", 8, 16)
        mock_ram.return_value = 16.0
        mock_nvidia.return_value = (True, "nvidia", 12.0)

        hw = detect_hardware()
        self.assertTrue(hw.gpu_available)
        self.assertEqual(hw.gpu_type, "nvidia")
        self.assertEqual(hw.gpu_vram_gb, 12.0)


if __name__ == "__main__":
    unittest.main()
