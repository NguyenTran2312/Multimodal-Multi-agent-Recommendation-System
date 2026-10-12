from __future__ import annotations


def invoke_structured(llm, schema, prompt: str, payload: dict):
    if hasattr(llm, "invoke_structured"):
        raw = llm.invoke_structured(schema, prompt, payload)
    elif hasattr(llm, "with_structured_output"):
        raw = llm.with_structured_output(schema).invoke(
            f"{prompt}\n\nInput:\n{payload}"
        )
    else:
        raise TypeError("LLM client must support invoke_structured() or with_structured_output()")
    if isinstance(raw, schema):
        return raw
    return schema.model_validate(raw)
