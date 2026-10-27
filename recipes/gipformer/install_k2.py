#!/usr/bin/env python3
"""Install the k2 wheel built for the torch already in this environment (no torch reinstall).

k2 wheels are compiled against one exact torch + CUDA version, listed at
https://k2-fsa.github.io/k2/cuda.html. This picks the newest one matching ``torch.__version__``,
``torch.version.cuda`` and the Python version, and pip-installs it with ``--no-deps``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
import urllib.request

INDEX = "https://k2-fsa.github.io/k2/cuda.html"


def main() -> None:
    import torch

    torch_version = torch.__version__.split("+")[0]
    cuda = torch.version.cuda
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"torch {torch_version}, CUDA {cuda}, {py}")
    if not cuda:
        raise SystemExit("this torch has no CUDA support; k2 needs a CUDA build of torch")

    html = urllib.request.urlopen(INDEX, timeout=120).read().decode()
    links = [urllib.parse.unquote(urllib.parse.urljoin(INDEX, h)) for h in re.findall(r'href="([^"]+?\.whl)"', html)]
    tag = f"+cuda{cuda}.torch{torch_version}-{py}-"
    match = [u for u in links if tag in u.rsplit("/", 1)[-1] and "x86_64" in u]
    if not match:
        near = sorted({re.sub(r"^.*\+", "+", u.rsplit("/", 1)[-1]).split("-")[0] for u in links if py in u})[-15:]
        raise SystemExit(
            f"no k2 wheel for torch {torch_version} + CUDA {cuda} + {py}.\n"
            f"Available builds (latest): {', '.join(near)}\n"
            "Use a torch version from this list in a separate conda env."
        )
    url = sorted(match, key=lambda u: u.rsplit("/", 1)[-1])[-1]
    print("installing", url.rsplit("/", 1)[-1])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", urllib.parse.quote(url, safe=":/")])
    import k2  # noqa: F401  (fails loudly if the wheel does not load)

    print("k2 OK")


if __name__ == "__main__":
    main()
