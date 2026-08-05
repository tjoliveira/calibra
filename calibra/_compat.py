"""Prompt formatting helpers (no external chat-template dependencies)."""


def format_mistral_prompt(system: str, user: str) -> str:
    """Format a Mistral-style instruction prompt.

    Args:
        system: System message.
        user: User turn content.

    Returns:
        Formatted prompt string ready for tokenisation.
    """
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
