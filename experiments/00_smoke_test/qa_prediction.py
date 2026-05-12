from __future__ import annotations

from contextflow.evaluation import parse_cacheblend_generation


def main() -> None:
    multiline_answer = "\n\nAnswer\nmore"
    assert parse_cacheblend_generation(multiline_answer) == "Answer"
    assert parse_cacheblend_generation("Yes, it is...") == "Yes"
    assert parse_cacheblend_generation("yes...") == "Yes"
    assert parse_cacheblend_generation("No, ...") == "No"
    assert parse_cacheblend_generation("no ...") == "No"
    assert parse_cacheblend_generation("Jane Austen\nbecause the passage says so") == "Jane Austen"

    print("qa_prediction smoke test passed")
    print(f"parsed: {parse_cacheblend_generation(multiline_answer)!r}")


if __name__ == "__main__":
    main()
