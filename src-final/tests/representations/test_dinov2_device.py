import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from stream_analysis.representations import (
    DinoV2ProviderError,
    LocalDinoV2Provider,
    checkpoint_sha256,
    resolve_device,
    source_tree_fingerprint,
)


class _CudaState:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _TorchState:
    def __init__(self, available: bool) -> None:
        self.cuda = _CudaState(available)


class DinoV2DevicePolicyTest(unittest.TestCase):
    def test_cpu_policy_is_explicit(self) -> None:
        self.assertEqual(resolve_device(_TorchState(True), "cpu"), ("cpu", None))

    def test_auto_prefers_cuda_and_warns_on_cpu_fallback(self) -> None:
        self.assertEqual(resolve_device(_TorchState(True), "auto"), ("cuda", None))
        self.assertEqual(
            resolve_device(_TorchState(False), "auto"),
            ("cpu", "DEVICE_FALLBACK_CPU"),
        )

    def test_unavailable_explicit_cuda_is_structured_error(self) -> None:
        with self.assertRaises(DinoV2ProviderError) as raised:
            resolve_device(_TorchState(False), "cuda")
        self.assertEqual(raised.exception.code, "DEVICE_UNAVAILABLE")

    def test_explicit_cuda_never_falls_back_to_cpu_or_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "hubconf.py").write_text("# local\n", encoding="utf-8")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
                expected_source_tree_fingerprint=source_tree_fingerprint(source),
                expected_checkpoint_size_bytes=checkpoint.stat().st_size,
                device_policy="cuda",
            )
            fake_torch = _TorchState(False)
            fake_torch.hub = Mock()
            with patch(
                "stream_analysis.representations.dinov2_provider._import_torch",
                return_value=fake_torch,
            ):
                with self.assertRaises(DinoV2ProviderError) as raised:
                    provider.prepare()

            self.assertEqual(raised.exception.code, "DEVICE_UNAVAILABLE")
            fake_torch.hub.load.assert_not_called()
            self.assertIsNone(provider.resolved_device)


if __name__ == "__main__":
    unittest.main()
