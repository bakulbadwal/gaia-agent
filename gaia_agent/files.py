"""Download and cache the file attached to a GAIA task.

Default source: the course API, GET {API_URL}/files/{task_id}.

Optional (GAIA_ATTACHMENTS_FROM_HUB=1, default off): when the course API has no file, fetch
THAT ONE attachment from the gated GAIA dataset on the Hugging Face Hub
(``2023/validation/<file_name>``). Only the attachment is ever requested, never the metadata
or any answer file, and a 401/403 (gated, no access for this token) returns None with a
clear message instead of raising.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path

import requests

from .config import API_URL, FILES_DIR

logger = logging.getLogger(__name__)

GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_ATTACHMENT_PREFIX = "2023/validation/"
# Extensions that GAIA uses for metadata / answers rather than attachments: never fetched.
_NEVER_FETCH = re.compile(r"(metadata|\.jsonl$|\.parquet$)", re.I)
last_message: str | None = None  # the most recent download problem, for logs and traces


def _safe_name(name: str) -> str:
    name = os.path.basename(name.strip().replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return name or "file"


def local_path_for(task_id: str, file_name: str, files_dir: Path | None = None) -> Path:
    return Path(files_dir or FILES_DIR) / f"{task_id}_{_safe_name(file_name)}"


def download_task_file(
    task_id: str,
    file_name: str | None,
    *,
    files_dir: str | os.PathLike | None = None,
    api_url: str = API_URL,
    force: bool = False,
    timeout: float = 60,
) -> str | None:
    """GET {API_URL}/files/{task_id} and save it as data/files/<task_id>_<file_name>.

    Returns the local path, or None when the task has no file or the server has none.
    Cached: an existing non-empty file is returned without a network call.
    """
    if not task_id or not file_name or not str(file_name).strip():
        return None
    dest = local_path_for(task_id, file_name, Path(files_dir) if files_dir else None)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(f"{api_url}/files/{task_id}", timeout=timeout, stream=True)
    if resp.status_code == 404:
        if hub_enabled():
            return download_from_hub(file_name, dest)
        return None
    resp.raise_for_status()
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(dest)
    return str(dest)


# --------------------------------------------------------------------------- optional HF Hub source


def hub_enabled() -> bool:
    return (os.environ.get("GAIA_ATTACHMENTS_FROM_HUB") or "0").strip().lower() in ("1", "true", "yes", "on")


def _note(msg: str) -> None:
    global last_message
    last_message = msg
    logger.warning(msg)
    print(msg, file=sys.stderr)


def download_from_hub(file_name: str, dest: str | os.PathLike, *, hf_hub_download=None) -> str | None:
    """Fetch one GAIA attachment from the gated dataset and copy it to ``dest``.

    Requests exactly ``2023/validation/<basename(file_name)>``; refuses anything that looks like
    metadata. Returns the local path, or None (with a message) on any failure, including
    401/403 when this HF token has no access to the gated dataset.
    """
    name = os.path.basename(str(file_name or "").strip().replace("\\", "/"))
    if not name or name in (".", "..") or _NEVER_FETCH.search(name):
        _note(f"GAIA hub attachment refused: {file_name!r} is not an attachment name.")
        return None
    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            _note("GAIA hub attachment unavailable: huggingface_hub is not installed.")
            return None
    filename = GAIA_ATTACHMENT_PREFIX + name
    try:
        cached = hf_hub_download(repo_id=GAIA_REPO_ID, repo_type="dataset", filename=filename)
    except Exception as e:  # noqa: BLE001 - every failure becomes None + a message
        status = getattr(getattr(e, "response", None), "status_code", None)
        kind = type(e).__name__
        if status in (401, 403) or kind in ("GatedRepoError", "RepositoryNotFoundError"):
            _note(
                f"GAIA hub attachment {filename!r}: access denied (HTTP {status or '401/403'}, {kind}). "
                "The dataset is gated: accept its terms on the Hub with the account behind HF_TOKEN. "
                "Continuing without the file."
            )
        elif status == 404 or kind == "EntryNotFoundError":
            _note(f"GAIA hub attachment {filename!r} does not exist (404). Continuing without the file.")
        else:
            _note(f"GAIA hub attachment {filename!r} failed: {kind}: {str(e)[:200]}. Continuing without the file.")
        return None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        _note(f"GAIA hub attachment {filename!r} was empty.")
        return None
    return str(dest)
