# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Resolve ACP ``session/request_permission`` requests to a chosen option.

Returns the ``option_id`` to select (an *allowed* outcome) or ``None`` to deny.
The bridge wraps the result as an ``AllowedOutcome``/``DeniedOutcome``; keeping
this layer SDK-free makes it unit-testable without ``acp``.
"""

_ALLOW_KINDS = ("allow_once", "allow_always")
_REJECT_KINDS = ("reject_once", "reject_always")

_AFFIRMATIVE = {"y", "yes", "allow", "allow_once", "allow_always", "ok", "approve"}


def _option_id_of_kind(options: list[dict], kinds: tuple[str, ...]) -> str | None:
    for o in options:
        if o.get("kind") in kinds:
            return o.get("option_id")
    return None


async def resolve_permission_option_id(
    policy: str,
    options: list[dict],
    tool_call: dict,
    context: object | None,
) -> str | None:
    """Return the ``option_id`` to allow, or ``None`` to deny.

    Args:
        policy: ``"ask"`` | ``"auto"`` | ``"deny"``.
        options: ACP permission options as dicts (``option_id``, ``kind``, ``name``).
        tool_call: The ``ToolCallUpdate`` dict (used to describe the action when asking).
        context: A ``ConversationContext`` (anything with an async ``input``) or ``None``.
    """
    allow_id = _option_id_of_kind(options, _ALLOW_KINDS)
    reject_id = _option_id_of_kind(options, _REJECT_KINDS)

    if policy == "auto":
        return allow_id
    if policy == "deny":
        return reject_id

    # policy == "ask"
    if context is None or not hasattr(context, "input"):
        return reject_id

    title = tool_call.get("title") or "an action"
    rendered = "\n".join(f"- {o.get('name') or o.get('kind')}" for o in options)
    answer = await context.input(f"Agent requests permission for: {title}\nOptions:\n{rendered}\nAllow? (yes/no)")
    if answer.strip().lower() in _AFFIRMATIVE:
        return allow_id
    return reject_id
