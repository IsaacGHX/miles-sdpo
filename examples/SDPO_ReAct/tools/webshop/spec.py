"""OpenAI function-call spec for the ``webshop_step`` tool (WebShop's
``WebAgentTextEnv-v0`` gym env, wrapped by tools/webshop/docker/server.py).

WebShop's action space is exactly two raw-text forms -- ``search[query]`` and
``click[button text]`` -- so this mirrors alfworld_step/cli_exec's single-
string-parameter shape rather than inventing separate search/click tools:
splitting them would just re-invent the same string the model already has to
produce, for no benefit.
"""

WEBSHOP_STEP_SPEC = {
    "type": "function",
    "function": {
        "name": "webshop_step",
        "description": (
            "Take one action in the WebShop online-shopping environment. The observation "
            "returned after each call is the current page text (search results, product "
            "page, or product options) plus the list of currently clickable buttons -- "
            "choose your next action from exactly two forms: 'search[query]' to search "
            "for products, or 'click[button text]' to click a button/link shown on the "
            "current page (e.g. 'click[Buy Now]', 'click[< Prev]', a product's ASIN, or an "
            "option like a color/size). The episode ends when you buy a product or the "
            "turn budget runs out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to take, e.g. 'search[wireless mouse]' or 'click[Buy Now]'.",
                },
            },
            "required": ["action"],
        },
    },
}

webshop_specs = [WEBSHOP_STEP_SPEC]
