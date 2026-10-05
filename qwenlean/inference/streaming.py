"""Defensive streaming state machine; no device or model dependency."""

from qwenlean.metrics.repetition import suffix_loop


def validate_eos_ids(ids, vocab_size=None):
    if not ids:
        raise ValueError("No EOS token metadata; refusing generation without a valid stop set")
    ids = set(ids)
    if any(
        type(i) is not int or i < 0 or (vocab_size is not None and i >= vocab_size) for i in ids
    ):
        raise ValueError("Invalid EOS token metadata")
    return ids


def consume_stream(
    iterator, *, eos_ids, max_tokens, thinking, closing, opening=None, guard=None, on_token=None
):
    eos_ids = validate_eos_ids(eos_ids)
    guard = guard or {}
    ids, response, error, evidence = [], None, None, None
    finish_reason = None
    in_reasoning = thinking
    try:
        for response in iterator:
            token = response.token
            if type(token) is not int or token < 0:
                raise ValueError("Stream emitted an invalid token ID")
            ids.append(token)
            finish_reason = getattr(response, "finish_reason", None)
            if on_token:
                on_token(len(ids))
            if token in eos_ids:
                break
            if token == closing:
                in_reasoning = False
            elif opening is not None and token == opening:
                in_reasoning = True
            if (
                guard.get("enabled", False)
                and in_reasoning
                and len(ids) % guard.get("check_every", 16) == 0
            ):
                evidence = suffix_loop(
                    ids,
                    **{
                        k: v
                        for k, v in guard.items()
                        if k in {"min_period", "max_period", "repeats"}
                    },
                )
                if evidence:
                    break
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            close = getattr(iterator, "close", None)
            if close:
                close()
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "iterator_close",
                "prior_error": error,
            }
    if error:
        reason = "cleanup_error" if error.get("stage") == "iterator_close" else "stream_error"
    elif evidence:
        reason = "runtime_loop_guard"
    elif not ids:
        reason = "empty_generation"
    elif ids[-1] in eos_ids:
        reason = "eos"
    elif finish_reason == "length":
        reason = "max_output_tokens" if len(ids) == max_tokens else "unexpected_length_stop"
    elif finish_reason == "stop":
        reason = "backend_stop_without_eos"
    elif finish_reason is not None:
        reason = "unknown_finish_reason"
    else:
        reason = "iterator_exhausted"
    return {
        "token_ids": ids,
        "last_response": response,
        "termination_reason": reason,
        "backend_finish_reason": finish_reason,
        "generation_error": error,
        "loop_guard_evidence": evidence,
    }


def validate_context_budget(prompt_tokens, output_budget, context_window):
    if context_window is None or context_window <= 0:
        raise ValueError("Missing/invalid model context-window metadata")
    if prompt_tokens + output_budget > context_window:
        raise ValueError(
            "Prompt plus output budget exceeds the original model context window; no truncation is performed"
        )
