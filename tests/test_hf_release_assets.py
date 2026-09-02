import base64
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit

import pytest

TOOLS_CI = Path(__file__).resolve().parents[1] / "tools" / "ci"
sys.path.insert(0, str(TOOLS_CI))

package_module = import_module("package_hf_assets")
publish_module = import_module("publish_hf_assets")
refs_module = import_module("hf_git_refs")
verify_module = import_module("verify_hf_assets")
MANIFEST_NAME = package_module.MANIFEST_NAME
package_assets = package_module.package_assets
create_tag_via_git = publish_module._create_tag_via_git
delete_tag_via_git = publish_module._delete_tag_via_git
run_git = publish_module._run_git
verify_assets = verify_module.verify_assets


class _GitHttpHandler(BaseHTTPRequestHandler):
    project_root: Path
    expected_authorization: str
    seen_authorizations: list[str | None]

    def log_message(self, _format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self._serve_git()

    def do_POST(self) -> None:
        self._serve_git()

    def _serve_git(self) -> None:
        authorization = self.headers.get("Authorization")
        self.seen_authorizations.append(authorization)
        if authorization != self.expected_authorization:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="test"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        parsed = urlsplit(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(content_length)
        env = os.environ.copy()
        env.update(
            {
                "CONTENT_LENGTH": str(content_length),
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "GIT_HTTP_EXPORT_ALL": "1",
                "GIT_PROJECT_ROOT": str(self.project_root),
                "PATH_INFO": parsed.path,
                "QUERY_STRING": parsed.query,
                "REMOTE_ADDR": self.client_address[0],
                "REQUEST_METHOD": self.command,
            }
        )
        result = subprocess.run(
            ["git", "http-backend"],
            input=request_body,
            capture_output=True,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            self.send_error(500, result.stderr.decode(errors="replace"))
            return

        separator = b"\r\n\r\n"
        if separator not in result.stdout:
            separator = b"\n\n"
        header_bytes, response_body = result.stdout.split(separator, 1)
        status = 200
        headers: list[tuple[str, str]] = []
        for line in header_bytes.decode().splitlines():
            key, value = line.split(":", 1)
            if key.lower() == "status":
                status = int(value.strip().split(" ", 1)[0])
            else:
                headers.append((key, value.strip()))

        self.send_response(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


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


def test_v0_3_contract_contains_supported_hand_assets_only() -> None:
    contract = json.loads((TOOLS_CI / "hf_assets_v0.3.json").read_text(encoding="utf-8"))
    sources = {entry["source"] for entry in contract["files"]}

    assert contract["release_series"] == "0.3"
    assert {
        "assets/SOMAHand.npz",
        "assets/MANO/SOMA_wrap_left.obj",
        "assets/MANO/SOMA_wrap_right.obj",
        "assets/MANO/base_hand_left.obj",
        "assets/MANO/base_hand_right.obj",
        "assets/images/soma-in-action.gif",
    } <= sources
    assert not any("UmeTrack" in source for source in sources)
    assert not any(source.startswith("tools/") for source in sources)
    # The SOMA Hand teaser is part of the combined soma-in-action.gif; no
    # separate hand render ships in the payload.
    hand_media = {
        source
        for source in sources
        if "hand" in Path(source).name.lower()
        and Path(source).suffix.lower() in {".gif", ".jpeg", ".jpg", ".mp4", ".png", ".webp"}
    }
    assert hand_media == set()


def test_top_level_exports_hand_layers() -> None:
    import soma

    assert soma.SOMAHandLayer.__module__ == "soma.hand.soma"
    assert soma.MANOLayer.__module__ == "soma.hand.mano"


def test_create_tag_via_git_uses_scoped_token_only_for_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str],
    ) -> object:
        calls.append((command, env))
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(publish_module.subprocess, "run", fake_run)

    create_tag_via_git("nvidia/SOMA-X", "v0.2.3", "a" * 40, "hf_test_token")

    assert len(calls) == 5
    assert calls[2][0][-2:] == ["origin", "a" * 40]
    assert "FETCH_HEAD" in calls[3][0]
    assert "hf_test_token" not in " ".join(calls[1][0])
    assert calls[-1][0][-1] == "refs/tags/v0.2.3"
    assert calls[-1][1]["GIT_TERMINAL_PROMPT"] == "0"
    encoded = base64.b64encode(b"hf_user:hf_test_token").decode()
    assert calls[-1][1]["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {encoded}"
    assert "hf_test_token" not in " ".join(calls[-1][0])


def test_delete_tag_via_git_uses_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str],
    ) -> object:
        calls.append((command, env))
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(publish_module.subprocess, "run", fake_run)

    delete_tag_via_git("nvidia/SOMA-X", "ci-oidc-123-1", "hf_test_token")

    assert len(calls) == 3
    assert calls[-1][0][-1] == ":refs/tags/ci-oidc-123-1"
    assert calls[-1][1]["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")


def test_git_failure_redacts_token(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = base64.b64encode(b"hf_user:hf_secret_token").decode()

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        env: dict[str, str],
    ) -> object:
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stderr": f"remote rejected hf_secret_token and {encoded}",
            },
        )()

    monkeypatch.setattr(publish_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"remote rejected \*\*\*") as exc_info:
        run_git(["push"], token="hf_secret_token")
    assert "hf_secret_token" not in str(exc_info.value)
    assert encoded not in str(exc_info.value)


def test_git_push_succeeds_against_basic_auth_http_server(tmp_path: Path) -> None:
    remote_root = tmp_path / "remotes"
    remote_root.mkdir()
    remote = remote_root / "model.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        capture_output=True,
        check=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(remote), "config", "http.receivepack", "true"],
        capture_output=True,
        check=True,
        text=True,
    )

    local = tmp_path / "local"
    subprocess.run(
        ["git", "init", "--quiet", str(local)],
        capture_output=True,
        check=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(local), "config", "user.name", "Test"],
        capture_output=True,
        check=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(local), "config", "user.email", "test@example.com"],
        capture_output=True,
        check=True,
        text=True,
    )
    (local / "README.md").write_text("test\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(local), "add", "README.md"],
        capture_output=True,
        check=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(local), "commit", "--quiet", "-m", "test"],
        capture_output=True,
        check=True,
        text=True,
    )

    token = "hf_test_token"
    expected_authorization = "Basic " + base64.b64encode(f"hf_user:{token}".encode()).decode()
    handler = type(
        "GitHttpHandler",
        (_GitHttpHandler,),
        {
            "project_root": remote_root,
            "expected_authorization": expected_authorization,
            "seen_authorizations": [],
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        run_git(
            [
                "-C",
                str(local),
                "push",
                f"http://127.0.0.1:{server.server_port}/model.git",
                "HEAD:refs/heads/main",
            ],
            token=token,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    remote_head = subprocess.run(
        ["git", "-C", str(remote), "rev-parse", "refs/heads/main"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    local_head = subprocess.run(
        ["git", "-C", str(local), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert remote_head == local_head
    assert handler.seen_authorizations
    assert set(handler.seen_authorizations) == {expected_authorization}


def test_smoke_git_auth_always_removes_created_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags: set[str] = set()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(refs_module, "_tag_names", lambda repo_id, token: set(tags))
    monkeypatch.setattr(
        refs_module,
        "_manifest_bytes",
        lambda repo_id, revision, token: b"manifest",
    )

    def fake_create(repo_id: str, tag: str, revision: str, token: str) -> None:
        tags.add(tag)
        calls.append(("create", tag))

    def fake_delete(repo_id: str, tag: str, token: str) -> None:
        tags.remove(tag)
        calls.append(("delete", tag))

    monkeypatch.setattr(refs_module, "_create_tag_via_git", fake_create)
    monkeypatch.setattr(refs_module, "_delete_tag_via_git", fake_delete)

    refs_module.smoke_git_auth(
        "nvidia/SOMA-X",
        "main",
        "ci-oidc-123-1",
        "hf_test_token",
    )

    assert calls == [("create", "ci-oidc-123-1"), ("delete", "ci-oidc-123-1")]
    assert not tags


def test_smoke_git_auth_cleans_up_after_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags: set[str] = set()

    monkeypatch.setattr(refs_module, "_tag_names", lambda repo_id, token: set(tags))
    monkeypatch.setattr(
        refs_module,
        "_manifest_bytes",
        lambda repo_id, revision, token: b"tag" if revision.startswith("ci-") else b"source",
    )
    monkeypatch.setattr(
        refs_module,
        "_create_tag_via_git",
        lambda repo_id, tag, revision, token: tags.add(tag),
    )
    monkeypatch.setattr(
        refs_module,
        "_delete_tag_via_git",
        lambda repo_id, tag, token: tags.remove(tag),
    )

    with pytest.raises(ValueError, match="does not resolve"):
        refs_module.smoke_git_auth(
            "nvidia/SOMA-X",
            "main",
            "ci-oidc-123-1",
            "hf_test_token",
        )
    assert not tags


def test_create_verified_release_tag_checks_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tags: set[str] = set()
    manifest = json.dumps({"release_tag": "v0.2.3"}).encode()

    monkeypatch.setattr(refs_module, "_tag_names", lambda repo_id, token: set(tags))
    monkeypatch.setattr(
        refs_module,
        "_manifest_bytes",
        lambda repo_id, revision, token: manifest,
    )
    monkeypatch.setattr(
        refs_module,
        "_create_tag_via_git",
        lambda repo_id, tag, revision, token: tags.add(tag),
    )

    refs_module.create_verified_release_tag(
        "nvidia/SOMA-X",
        "a" * 40,
        "v0.2.3",
        "hf_test_token",
    )

    assert tags == {"v0.2.3"}


def test_create_verified_release_tag_rejects_wrong_release_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        refs_module,
        "_manifest_bytes",
        lambda repo_id, revision, token: json.dumps(
            {"release_tag": "v0.2.2"}
        ).encode(),
    )

    with pytest.raises(ValueError, match="does not match"):
        refs_module.create_verified_release_tag(
            "nvidia/SOMA-X",
            "a" * 40,
            "v0.2.3",
            "hf_test_token",
        )
