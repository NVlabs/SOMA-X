import json
import sys
from importlib import import_module
from pathlib import Path

import pytest

TOOLS_CI = Path(__file__).resolve().parents[1] / "tools" / "ci"
sys.path.insert(0, str(TOOLS_CI))

package_module = import_module("package_hf_assets")
publish_module = import_module("publish_hf_assets")
verify_module = import_module("verify_hf_assets")
MANIFEST_NAME = package_module.MANIFEST_NAME
package_assets = package_module.package_assets
create_tag_via_git = publish_module._create_tag_via_git
run_git = publish_module._run_git
verify_assets = verify_module.verify_assets


def _write_repo(root: Path, content: bytes = b"asset-data") -> Path:
    (root / "soma").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "setup.cfg").write_text("[metadata]\nversion = 0.2.2\n", encoding="utf-8")
    (root / "soma" / "__init__.py").write_text('__version__ = "0.2.2"\n', encoding="utf-8")
    (root / "assets" / "model.bin").write_bytes(content)
    contract = root / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_series": "0.2",
                "files": [{"source": "assets/model.bin", "path": "model.bin"}],
            }
        ),
        encoding="utf-8",
    )
    return contract


def _package(tmp_path: Path, content: bytes = b"asset-data") -> Path:
    root = tmp_path / "repo"
    contract = _write_repo(root, content)
    stage = tmp_path / "stage"
    package_assets(root, contract, stage, "v0.2.2", "a" * 40)
    return stage


def test_package_and_verify_assets(tmp_path: Path) -> None:
    stage = _package(tmp_path)

    manifest = verify_assets(stage)

    assert manifest["release_tag"] == "v0.2.2"
    assert manifest["public_commit"] == "a" * 40
    assert manifest["files"][0]["path"] == "model.bin"
    assert (stage / MANIFEST_NAME).is_file()


def test_package_rejects_unresolved_lfs_pointer(tmp_path: Path) -> None:
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64
    root = tmp_path / "repo"
    contract = _write_repo(root, pointer)

    with pytest.raises(ValueError, match="unresolved Git LFS pointer"):
        package_assets(root, contract, tmp_path / "stage", "v0.2.2", "a" * 40)


def test_package_rejects_release_series_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    contract = _write_repo(root)

    with pytest.raises(ValueError, match="does not match contract series"):
        package_assets(root, contract, tmp_path / "stage", "v0.3.0", "a" * 40)


@pytest.mark.parametrize("mutation", ["tamper", "extra"])
def test_verify_rejects_content_outside_manifest(tmp_path: Path, mutation: str) -> None:
    stage = _package(tmp_path)
    if mutation == "tamper":
        (stage / "model.bin").write_bytes(b"changed")
        expected = "Size mismatch"
    else:
        (stage / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        expected = "file set mismatch"

    with pytest.raises(ValueError, match=expected):
        verify_assets(stage)


def test_verify_allows_hub_gitattributes(tmp_path: Path) -> None:
    stage = _package(tmp_path)
    (stage / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")

    verify_assets(stage)


def test_verify_only_allows_download_cache_when_requested(tmp_path: Path) -> None:
    stage = _package(tmp_path)
    metadata = stage / ".cache" / "huggingface" / "download" / "model.bin.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("metadata", encoding="utf-8")

    with pytest.raises(ValueError, match="file set mismatch"):
        verify_assets(stage)

    verify_assets(stage, allow_download_cache=True)


def test_model_card_has_structured_hub_metadata() -> None:
    from huggingface_hub import ModelCard

    model_card = ModelCard.load(Path(__file__).resolve().parents[1] / "docs" / "model_card.md")

    assert model_card.data.license == "apache-2.0"
    assert "soma-x" in model_card.data.tags


def test_create_tag_via_git_uses_scoped_token_only_for_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> object:
        calls.append(command)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(publish_module.subprocess, "run", fake_run)

    create_tag_via_git("nvidia/SOMA-X", "v0.2.3", "a" * 40, "hf_test_token")

    assert len(calls) == 5
    assert calls[2][-2:] == ["origin", "a" * 40]
    assert "hf_test_token" not in " ".join(calls[1])
    assert "http.extraHeader=Authorization: Bearer hf_test_token" in calls[-1]
    assert calls[-1][-1] == "refs/tags/v0.2.3"


def test_git_failure_redacts_token(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> object:
        return type(
            "Result",
            (),
            {"returncode": 1, "stderr": "remote rejected hf_secret_token"},
        )()

    monkeypatch.setattr(publish_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"remote rejected \*\*\*") as exc_info:
        run_git(["push"], token="hf_secret_token")
    assert "hf_secret_token" not in str(exc_info.value)
