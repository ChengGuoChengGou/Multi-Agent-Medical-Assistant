import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core import PromptRegistry


def main():
    registry = PromptRegistry()

    query_prompt = registry.get("rag.query_expansion")
    rendered_query_prompt = query_prompt.render(query="brain tumor MRI diagnosis")
    assert "brain tumor MRI diagnosis" in rendered_query_prompt
    assert query_prompt.metadata()["prompt_id"] == "rag.query_expansion"
    assert query_prompt.metadata()["prompt_version"] == "v1"

    response_prompt = registry.get("rag.response_generation")
    rendered_response_prompt = response_prompt.render(
        query="What are signs of COVID-19 on X-ray?",
        context="Ground-glass opacity is mentioned in the context.",
        chat_history="",
    )
    assert "What are signs of COVID-19 on X-ray?" in rendered_response_prompt
    assert "Ground-glass opacity" in rendered_response_prompt

    print("Prompt registry checks passed")


if __name__ == "__main__":
    main()
