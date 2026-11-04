"""Replace a file whole, so a crash or a full disk never leaves half of it behind."""

import os
import shutil
import uuid
from pathlib import Path


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write UTF-8 *text* to a temp file beside *path*, then rename it over *path*.

    Matches ``Path.write_text`` otherwise: a symlink keeps pointing at the replaced file, an existing file keeps
    its mode, a new one gets the umask's, and a non-regular path (``/dev/stdout``, a pipe) is written directly.
    """
    target = Path(path).resolve()
    if target.exists() and not target.is_file():
        with target.open("w", encoding="utf-8") as handle:
            handle.write(text)
        return
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        if target.exists():
            shutil.copymode(target, temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
