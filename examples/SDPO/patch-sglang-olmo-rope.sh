#!/bin/bash
# Idempotent patch for a sglang bug that crashes Olmo2/Olmo3 model loading.
#
# Bug: sglang's olmo2.py reads `config.rope_parameters["rope_theta"]`, but
# sglang's OWN Olmo3Config (sglang/srt/configs/olmo3.py) builds
# `rope_parameters` from HF's `rope_scaling` dict WITHOUT copying
# `rope_theta` into it -- `rope_theta` only ever lands as a separate
# top-level `config.rope_theta` attribute (confirmed: 500000, via
# ModelConfig(...).hf_config on this image). So `rope_parameters["rope_theta"]`
# KeyErrors even though `config.rope_theta` itself is set -> crashes
# ModelRunner.load_model(), killing the SGLang scheduler before any rollout
# can start. Confirmed on both radixark/miles:dev-cu12-202607040446 and
# radixark/miles:latest-cu12 -- this is a sglang-internal config-construction
# gap, not an image-specific regression, and not fixed by pulling a newer
# image tag.
#
# Fix: prefer the top-level `config.rope_theta` attribute (always present on
# this Olmo3Config), falling back to the `rope_parameters` dict lookup only
# if that attribute is absent (covers older sglang Olmo2Config shapes that
# genuinely only had the dict). Runs inside the container; safe to run
# repeatedly (grep-guarded), so it survives container rebuilds when invoked
# from the launcher.
set -eu

F=/sgl-workspace/sglang/python/sglang/srt/models/olmo2.py

if [ ! -f "$F" ]; then
    echo "patch-sglang-olmo-rope: $F not found, skipping" >&2
    exit 0
fi

if grep -q "# sdpo-patch-rope" "$F"; then
    echo "patch-sglang-olmo-rope: already applied"
    exit 0
fi

python - "$F" <<'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()
pat = re.compile(r'self\.rope_theta = config\.rope_parameters\["rope_theta"\]')
def repl(m):
    return (
        'self.rope_theta = (\n'
        '            config.rope_theta\n'
        '            if getattr(config, "rope_theta", None) is not None\n'
        '            else config.rope_parameters["rope_theta"]\n'
        '        )  # sdpo-patch-rope'
    )
src2, n = pat.subn(repl, src)
if n == 0:
    print("patch-sglang-olmo-rope: target pattern not found (sglang version changed?) — leaving file untouched", file=sys.stderr)
    sys.exit(0)
open(path, "w").write(src2)
print(f"patch-sglang-olmo-rope: patched {n} site(s)")
PY
