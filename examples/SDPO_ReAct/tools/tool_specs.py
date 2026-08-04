"""Back-compat shim: the code_interpreter-only spec list this module used to
define directly now lives in tools/code/spec.py (see that subpackage's
docstring for why tools are split one-per-capability). Kept as a thin
re-export so existing ``--generate-tool-specs-path
examples.SDPO_ReAct.tools.tool_specs.tool_specs`` launcher flags and
``from examples.SDPO_ReAct.tools.tool_specs import ...`` imports keep working
unchanged.
"""

from examples.SDPO_ReAct.tools.cli.spec import CLI_EXEC_SPEC
from examples.SDPO_ReAct.tools.code.spec import CODE_INTERPRETER_SPEC

# The active tool set for the single-tool (code_interpreter-only) base
# version -- see generate_with_tools.py / run-qwen2.5-7B-sdpo-react-dapo-math.sh.
tool_specs = [CODE_INTERPRETER_SPEC]
