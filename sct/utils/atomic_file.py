"""Replace a file whole, so a crash or a full disk never leaves half of it behind."""

import errno
import os
import shutil
import uuid
from pathlib import Path


def _write_in_place(path: str | Path, text: str) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write UTF-8 *text* to a temp file beside *path*, then rename it over *path*.

    Matches ``Path.write_text`` otherwise: a symlink keeps pointing at the replaced file, an existing file keeps
    its mode, a new one gets the umask's, and a read-only file raises ``PermissionError``. A non-regular path
    (a pipe, a device) or a place that cannot hold the temp file (``/dev/stdout``, a read-only folder, a name
    too long for the temp suffix) is written directly, without the atomic guarantee.
    """
    target = Path(path).resolve()
    if target.exists() and not (target.is_file() and os.access(target, os.W_OK)):
        _write_in_place(path, text)
        return
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except OSError as error:
        # On a full disk a direct write would truncate the old file and then fail.
        if error.errno in (errno.ENOSPC, errno.EDQUOT):
            raise
        _write_in_place(path, text)
        return
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        if target.exists():
            shutil.copymode(target, temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
