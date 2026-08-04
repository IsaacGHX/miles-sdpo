"""Back-compat shim: the code_interpreter-only executor this module used to
implement directly now lives in tools/code/client.py (see that subpackage's
docstring for why tools are split one-per-capability). Kept as a thin
re-export so existing ``--generate-execute-tool-function-path
examples.SDPO_ReAct.tools.tool_client.execute_tool`` launcher flags and
``from examples.SDPO_ReAct.tools.tool_client import execute_tool`` imports
(e.g. generate_with_tools.py) keep working unchanged.
"""

from examples.SDPO_ReAct.tools.code.client import execute_tool

__all__ = ["execute_tool"]
