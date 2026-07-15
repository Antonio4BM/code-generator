"""Tests for candidate file tree and file preview endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from codegen_workflow.api.service import _MAX_CANDIDATE_PREVIEW_BYTES


def _start_paused(client: TestClient) -> str:
    """Start a mocked workflow and stop at the coder human gate."""
    response = client.post("/run-ticket", json={"ticket": "Build a hello CLI"})
    assert response.status_code == 202
    body = response.json()
    assert body["interrupt"]["gate"] == "coder"
    return body["workflow_id"]


def test_list_candidate_files(client: TestClient) -> None:
    """Paused workflows expose the generated candidate file tree."""
    workflow_id = _start_paused(client)
    response = client.get(f"/runs/{workflow_id}/files")
    assert response.status_code == 200
    body = response.json()
    assert body["workflow_id"] == workflow_id
    assert body["files"] == ["README.md", "hello.py"]


def test_read_candidate_file(client: TestClient) -> None:
    """Clicking a listed path returns UTF-8 file contents."""
    workflow_id = _start_paused(client)
    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "hello.py"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "hello.py"
    assert body["encoding"] == "utf-8"
    assert body["content"] == "print('hello')\n"
    assert body["size_bytes"] == len("print('hello')\n".encode("utf-8"))


def test_read_missing_candidate_file(client: TestClient) -> None:
    """Unknown relative paths return 404."""
    workflow_id = _start_paused(client)
    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "missing.py"},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "candidate_file_not_found"


def test_path_traversal_rejected(client: TestClient) -> None:
    """Parent-directory traversal cannot escape candidate/."""
    workflow_id = _start_paused(client)
    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "../secrets.txt"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "file_path_violation"


def test_absolute_path_rejected(client: TestClient) -> None:
    """Absolute paths are rejected."""
    workflow_id = _start_paused(client)
    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "/etc/passwd"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "file_path_violation"


def test_binary_file_rejected(client: TestClient) -> None:
    """Non-UTF-8 files cannot be previewed."""
    workflow_id = _start_paused(client)
    service = client.app.state.workflow_service
    root = service._candidate_root(workflow_id)
    (root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")

    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "blob.bin"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "candidate_file_unreadable"


def test_oversized_file_rejected(client: TestClient) -> None:
    """Files larger than the preview cap are rejected."""
    workflow_id = _start_paused(client)
    service = client.app.state.workflow_service
    root = service._candidate_root(workflow_id)
    (root / "huge.txt").write_text(
        "x" * (_MAX_CANDIDATE_PREVIEW_BYTES + 1),
        encoding="utf-8",
    )

    response = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "huge.txt"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "candidate_file_unreadable"


def test_unknown_workflow_files(client: TestClient) -> None:
    """Unknown workflows return 404 for file endpoints."""
    workflow_id = "00000000-0000-0000-0000-000000000000"
    listed = client.get(f"/runs/{workflow_id}/files")
    assert listed.status_code == 404
    content = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "hello.py"},
    )
    assert content.status_code == 404


def test_files_readable_while_paused_and_after_approve_path(
    client: TestClient,
) -> None:
    """Files remain listable after progressing past the coder gate."""
    workflow_id = _start_paused(client)
    listed = client.get(f"/runs/{workflow_id}/files")
    assert listed.status_code == 200
    assert "hello.py" in listed.json()["files"]

    resumed = client.post(
        f"/runs/{workflow_id}/decision",
        json={"decision": "approve"},
    )
    assert resumed.status_code == 202
    assert resumed.json()["interrupt"]["gate"] == "reviewer"

    listed_again = client.get(f"/runs/{workflow_id}/files")
    assert listed_again.status_code == 200
    content = client.get(
        f"/runs/{workflow_id}/files/content",
        params={"path": "README.md"},
    )
    assert content.status_code == 200
    assert content.json()["content"] == "# hello\n"
