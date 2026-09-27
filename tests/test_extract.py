"""XML-tag extraction helpers (llm_utils.extract_tag / extract_all_tags)."""

from __future__ import annotations

from llm_utils import extract_all_tags, extract_tag


def test_extract_tag_basic():
    assert extract_tag("<a>hello</a>", "a") == "hello"


def test_extract_tag_strips_whitespace():
    assert extract_tag("<a>\n  hi \n</a>", "a") == "hi"


def test_extract_tag_is_case_insensitive():
    assert extract_tag("<Agent>sales</Agent>", "agent") == "sales"


def test_extract_tag_spans_newlines():
    assert extract_tag("<x>line1\nline2</x>", "x") == "line1\nline2"


def test_extract_tag_missing_returns_default():
    assert extract_tag("no tags here", "a", default="fallback") == "fallback"
    assert extract_tag("no tags here", "a") is None


def test_extract_tag_returns_first_only():
    assert extract_tag("<a>one</a><a>two</a>", "a") == "one"


def test_extract_all_tags_returns_every_occurrence():
    assert extract_all_tags("<a>one</a><a>two</a>", "a") == ["one", "two"]


def test_extract_all_tags_empty_when_absent():
    assert extract_all_tags("nothing", "a") == []
