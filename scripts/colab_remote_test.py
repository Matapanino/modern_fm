"""On-VM driver for the CUDA validation loop (run via ``colab exec -f``).

The Colab CLI reads this file locally and executes its contents in the remote
GPU kernel. It expects the committed tree at ``/content/modern_fm.tar.gz``
(see ``scripts/colab_gpu_test.sh``). It then:

  1. extracts the repo to ``/content/modern_fm``,
  2. installs rustup + maturin + pytest,
  3. builds the extension with ``--features cuda-backend``
     (``MATURIN_PEP517_ARGS``; cudarc dlopens the driver, no toolkit needed),
  4. asserts ``_backend.has_cuda()``,
  5. runs ``tests/test_cuda_parity.py`` (must pass, not skip) and the full
     suite, then ``benchmarks/bench_cuda.py`` (``--quick`` unless the
     uploaded ``/content/modernfm_full_bench`` flag file says ``1``),
  6. writes everything to ``/content/modernfm_gpu_report.md`` — the report the
     runbook asks to paste into the CUDA PR.
"""

import os
import subprocess
import sys
import tarfile

REPO = "/content/modern_fm"
TARBALL = "/content/modern_fm.tar.gz"
REPORT = "/content/modernfm_gpu_report.md"
CARGO_BIN = os.path.expanduser("~/.cargo/bin")
ENV = {**os.environ, "PATH": f"{CARGO_BIN}:{os.environ['PATH']}"}
sections = []


def sh(cmd, check=True, shell=False, cwd=REPO):
    show = cmd if shell else " ".join(cmd)
    print("+", show, flush=True)
    proc = subprocess.run(cmd, shell=shell, cwd=cwd, env=ENV, capture_output=True, text=True)
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    print(out[-4000:], flush=True)
    if check and proc.returncode != 0:
        sections.append((f"FAILED: {show}", out))
        write_report(ok=False)
        raise SystemExit(f"command failed: {show}")
    return out


def write_report(ok):
    with open(REPORT, "w") as f:
        f.write(f"# modern_fm CUDA validation — {'PASS' if ok else 'FAIL'}\n\n")
        for title, body in sections:
            f.write(f"## {title}\n\n```\n{body.strip()}\n```\n\n")
    print(f"report written to {REPORT}", flush=True)


os.makedirs(REPO, exist_ok=True)
with tarfile.open(TARBALL, "r:gz") as tf:
    tf.extractall(REPO)
print(f"extracted tree to {REPO}", flush=True)

sections.append(("GPU / driver", sh(["nvidia-smi"], cwd="/content")))

if not os.path.exists(f"{CARGO_BIN}/cargo"):
    sh(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
        " | sh -s -- -y --profile minimal --default-toolchain stable",
        shell=True,
        cwd="/content",
    )
sections.append(("rust toolchain", sh(["cargo", "--version"], cwd="/content")))

sh([sys.executable, "-m", "pip", "install", "-q", "maturin", "pytest"], cwd="/content")

ENV["MATURIN_PEP517_ARGS"] = "--features cuda-backend"
build_out = sh([sys.executable, "-m", "pip", "install", "-v", "."])
sections.append(("build (--features cuda-backend)", build_out[-1500:]))

probe = "from modern_fm import _backend; print('has_cuda:', _backend.has_cuda())"
has_cuda = sh([sys.executable, "-c", probe], cwd="/content")
sections.append(("has_cuda()", has_cuda))
if "has_cuda: True" not in has_cuda:
    write_report(ok=False)
    raise SystemExit("has_cuda() is False — build or driver problem")

parity = sh([sys.executable, "-m", "pytest", "tests/test_cuda_parity.py", "-q", "-rs"])
sections.append(("tests/test_cuda_parity.py (must pass, not skip)", parity))

full = sh([sys.executable, "-m", "pytest", "-q"])
sections.append(("full pytest -q", full[-1500:]))

# The flag arrives as a file (colab exec does not forward the local env).
try:
    with open("/content/modernfm_full_bench") as f:
        full_bench = f.read().strip() == "1"
except OSError:
    full_bench = os.environ.get("MODERNFM_FULL_BENCH") == "1"
bench_args = [] if full_bench else ["--quick"]
bench = sh([sys.executable, "benchmarks/bench_cuda.py", *bench_args])
sections.append((f"bench_cuda.py {' '.join(bench_args)} (transfer-inclusive)", bench))

write_report(ok=True)
print("VALIDATION PASS", flush=True)
