import os
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stream_analysis.representations import (
    DINO_EMBEDDING_DIMENSION,
    DinoV2ProviderError,
    DinoV2ProviderOutput,
    LocalDinoV2Provider,
    checkpoint_sha256,
    source_tree_fingerprint,
    source_tree_manifest_bytes,
)


def make_assets(root: Path) -> tuple[Path, Path, str, str, int]:
    source = root / "source"
    source.mkdir()
    (source / "hubconf.py").write_text("# local hub entry\n", encoding="utf-8")
    package = source / "pkg"
    package.mkdir()
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    checkpoint = root / "model.pth"
    checkpoint.write_bytes(b"deterministic-checkpoint")
    return (
        source,
        checkpoint,
        checkpoint_sha256(checkpoint),
        source_tree_fingerprint(source),
        checkpoint.stat().st_size,
    )


class _FakeModel:
    def __init__(self) -> None:
        self.loaded = False
        self.device = None
        self.eval_called = False

    def load_state_dict(self, state, strict=True):
        self.loaded = bool(state) and strict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class DinoV2ProviderAssetTest(unittest.TestCase):
    def test_provider_output_rejects_float64_without_implicit_conversion(self) -> None:
        embedding = np.full(
            (1, DINO_EMBEDDING_DIMENSION),
            1.0 / np.sqrt(DINO_EMBEDDING_DIMENSION),
            dtype=np.float64,
        )

        with self.assertRaisesRegex(TypeError, "dtype must be float32"):
            DinoV2ProviderOutput(embeddings=embedding, runtime_details={})

    def test_source_tree_manifest_is_canonical_and_excludes_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "z").mkdir(parents=True)
            (source / "é").mkdir()
            (source / "z" / "b.py").write_bytes(b"b")
            (source / "é" / "a.py").write_bytes(b"a")
            cache = source / "__pycache__"
            cache.mkdir()
            (cache / "ignored.pyc").write_bytes(b"ignored")
            (source / "also_ignored.pyc").write_bytes(b"ignored")

            manifest = source_tree_manifest_bytes(source)
            lines = manifest.decode("utf-8").split("\n")

            self.assertFalse(manifest.endswith(b"\n"))
            self.assertEqual(lines, sorted(lines, key=lambda line: line.split(":", 1)[0].encode("utf-8")))
            self.assertEqual(len(lines), 2)
            self.assertFalse(any("pycache" in line or ".pyc" in line for line in lines))
            self.assertEqual(source_tree_fingerprint(source), source_tree_fingerprint(source))

    def test_provider_is_lazy_and_reports_missing_assets_on_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = LocalDinoV2Provider(
                source_dir=root / "missing-source",
                checkpoint_path=root / "missing.pth",
                expected_checkpoint_sha256="0" * 64,
                expected_source_tree_fingerprint="1" * 64,
                device_policy="cpu",
            )
            self.assertIsNone(provider.resolved_device)
            with self.assertRaises(DinoV2ProviderError) as raised:
                provider.prepare()
            self.assertEqual(raised.exception.code, "MODEL_SOURCE_MISSING")

    def test_missing_checkpoint_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "hubconf.py").write_text("# local\n", encoding="utf-8")
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=root / "missing.pth",
                expected_checkpoint_sha256="0" * 64,
                expected_source_tree_fingerprint=source_tree_fingerprint(source),
                device_policy="cpu",
            )
            with self.assertRaises(DinoV2ProviderError) as raised:
                provider.validate_assets()
            self.assertEqual(raised.exception.code, "MODEL_WEIGHTS_MISSING")

    def test_size_hash_and_tree_mismatch_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, checkpoint, checkpoint_hash, tree_hash, size = make_assets(Path(tmp))
            cases = (
                ({"expected_checkpoint_size_bytes": size + 1}, "CHECKPOINT_SIZE_MISMATCH"),
                ({"expected_checkpoint_sha256": "0" * 64}, "CHECKPOINT_HASH_MISMATCH"),
                ({"expected_source_tree_fingerprint": "1" * 64}, "SOURCE_TREE_FINGERPRINT_MISMATCH"),
            )
            for overrides, code in cases:
                with self.subTest(code=code):
                    values = {
                        "expected_checkpoint_sha256": checkpoint_hash,
                        "expected_source_tree_fingerprint": tree_hash,
                        "expected_checkpoint_size_bytes": size,
                    }
                    values.update(overrides)
                    provider = LocalDinoV2Provider(
                        source_dir=source,
                        checkpoint_path=checkpoint,
                        device_policy="cpu",
                        **values,
                    )
                    with self.assertRaises(DinoV2ProviderError) as raised:
                        provider.validate_assets()
                    self.assertEqual(raised.exception.code, code)

    def test_prepare_uses_only_local_hub_source_and_does_not_change_torch_home(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            source, checkpoint, checkpoint_hash, tree_hash, size = make_assets(Path(tmp))
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_source_tree_fingerprint=tree_hash,
                expected_checkpoint_size_bytes=size,
                device_policy="cpu",
            )
            fake_model = _FakeModel()
            torch_home_before = os.environ.get("TORCH_HOME")
            with patch.object(torch.hub, "load", return_value=fake_model) as hub_load, patch.object(
                torch,
                "load",
                return_value={"weight": object()},
            ):
                provider.prepare()

            args, kwargs = hub_load.call_args
            self.assertEqual(Path(args[0]), source.resolve())
            self.assertNotIn("facebookresearch", args[0])
            self.assertEqual(args[1], "dinov2_vits14")
            self.assertEqual(kwargs, {"source": "local", "pretrained": False})
            self.assertEqual(os.environ.get("TORCH_HOME"), torch_home_before)
            self.assertTrue(fake_model.loaded)
            self.assertTrue(fake_model.eval_called)
            self.assertEqual(fake_model.device, "cpu")

    def test_prepare_blocks_network_attempt_from_local_hubconf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "hubconf.py").write_text(
                "from urllib.request import urlopen\n"
                "def dinov2_vits14(pretrained=False):\n"
                "    urlopen('https://example.invalid/model.pth')\n",
                encoding="utf-8",
            )
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"unused-checkpoint")
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_sha256(checkpoint),
                expected_source_tree_fingerprint=source_tree_fingerprint(source),
                expected_checkpoint_size_bytes=checkpoint.stat().st_size,
                device_policy="cpu",
            )

            with patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(DinoV2ProviderError) as raised:
                    provider.prepare()

            self.assertEqual(raised.exception.code, "NETWORK_ACCESS_BLOCKED")
            self.assertEqual(raised.exception.metadata["api"], "urllib.request.urlopen")
            urlopen.assert_not_called()

    def test_forward_blocks_network_and_restores_patched_apis(self) -> None:
        import torch

        class NetworkModel(torch.nn.Module):
            def forward(self, tensor):
                urllib.request.urlopen("https://example.invalid/inference")
                return torch.zeros(
                    (tensor.shape[0], DINO_EMBEDDING_DIMENSION),
                    dtype=torch.float32,
                    device=tensor.device,
                )

        with tempfile.TemporaryDirectory() as tmp:
            source, checkpoint, checkpoint_hash, tree_hash, size = make_assets(Path(tmp))
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_source_tree_fingerprint=tree_hash,
                expected_checkpoint_size_bytes=size,
                device_policy="cpu",
            )
            model = NetworkModel()
            socket_create_connection = socket.create_connection
            batch = np.zeros((1, 3, 8, 8), dtype=np.float32)

            with patch.object(torch.hub, "load", return_value=model), patch.object(
                torch,
                "load",
                return_value=model.state_dict(),
            ), patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(DinoV2ProviderError) as raised:
                    provider.embed_batch(batch)

                self.assertIs(urllib.request.urlopen, urlopen)
                self.assertIs(socket.create_connection, socket_create_connection)

            self.assertEqual(raised.exception.code, "NETWORK_ACCESS_BLOCKED")
            self.assertEqual(raised.exception.metadata["api"], "urllib.request.urlopen")
            urlopen.assert_not_called()

    def test_prepare_restores_original_torch_home_state_before_error(self) -> None:
        import torch

        original_marker = object()
        original_value = os.environ.get("TORCH_HOME", original_marker)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source, checkpoint, checkpoint_hash, tree_hash, size = make_assets(Path(tmp))
                for initial_value in (None, "original-torch-home"):
                    with self.subTest(initial_value=initial_value):
                        if initial_value is None:
                            os.environ.pop("TORCH_HOME", None)
                        else:
                            os.environ["TORCH_HOME"] = initial_value
                        fake_model = _FakeModel()

                        def mutating_hub_load(*args, **kwargs):
                            del args, kwargs
                            os.environ["TORCH_HOME"] = "mutated-torch-home"
                            return fake_model

                        provider = LocalDinoV2Provider(
                            source_dir=source,
                            checkpoint_path=checkpoint,
                            expected_checkpoint_sha256=checkpoint_hash,
                            expected_source_tree_fingerprint=tree_hash,
                            expected_checkpoint_size_bytes=size,
                            device_policy="cpu",
                        )
                        with patch.object(
                            torch.hub,
                            "load",
                            side_effect=mutating_hub_load,
                        ), patch.object(torch, "load", return_value={"weight": object()}):
                            with self.assertRaises(DinoV2ProviderError) as raised:
                                provider.prepare()

                        self.assertEqual(raised.exception.code, "TORCH_HOME_MUTATION_DETECTED")
                        self.assertTrue(raised.exception.metadata["restored"])
                        if initial_value is None:
                            self.assertNotIn("TORCH_HOME", os.environ)
                        else:
                            self.assertEqual(os.environ["TORCH_HOME"], initial_value)
        finally:
            if original_value is original_marker:
                os.environ.pop("TORCH_HOME", None)
            else:
                os.environ["TORCH_HOME"] = original_value

    def test_real_tensor_contract_is_float32_normalized_and_batch_stable(self) -> None:
        import torch

        class DeterministicModel(torch.nn.Module):
            def forward(self, tensor):
                means = tensor.mean(dim=(2, 3))
                output = torch.zeros(
                    (tensor.shape[0], DINO_EMBEDDING_DIMENSION),
                    dtype=torch.float32,
                    device=tensor.device,
                )
                output[:, :3] = means
                output[:, 3] = 1.0
                return output

        with tempfile.TemporaryDirectory() as tmp:
            source, checkpoint, checkpoint_hash, tree_hash, size = make_assets(Path(tmp))
            provider = LocalDinoV2Provider(
                source_dir=source,
                checkpoint_path=checkpoint,
                expected_checkpoint_sha256=checkpoint_hash,
                expected_source_tree_fingerprint=tree_hash,
                expected_checkpoint_size_bytes=size,
                device_policy="cpu",
                batch_size=2,
            )
            model = DeterministicModel()
            with patch.object(torch.hub, "load", return_value=model), patch.object(
                torch,
                "load",
                return_value=model.state_dict(),
            ):
                batch = np.stack(
                    (
                        np.zeros((3, 8, 8), dtype=np.float32),
                        np.ones((3, 8, 8), dtype=np.float32),
                    )
                )
                together = provider.embed_batch(batch).embeddings
                singleton = provider.embed_batch(batch[:1]).embeddings[0]

            self.assertEqual(together.shape, (2, DINO_EMBEDDING_DIMENSION))
            self.assertEqual(together.dtype, np.float32)
            self.assertTrue(np.isfinite(together).all())
            np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(together[0], singleton, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
