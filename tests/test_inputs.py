"""Payload builder tests (pandas/CSV/JSON/JSONL/schema)."""

import io
import json

import pytest


def test_from_dataframe_converts_and_cleans():
    pd = pytest.importorskip("pandas")
    import datetime

    from isis_report_lib.inputs import from_dataframe

    df = pd.DataFrame(
        {
            "product": ["Pen", None],
            "total": [10.5, float("nan")],
            "when": [pd.Timestamp("2024-01-01"), pd.NaT],
            "active": [True, False],
        }
    )
    payload = from_dataframe(df, meta={"title": "T"})
    assert payload["meta"] == {"title": "T"}
    rows = payload["data"]
    assert rows[0]["product"] == "Pen"
    assert rows[1]["product"] is None
    assert rows[1]["total"] is None
    assert rows[0]["when"] == datetime.datetime(2024, 1, 1)
    assert rows[1]["when"] is None
    assert "layout" not in payload
    payload = from_dataframe(df, layout={"product": {"label": "Product"}})
    assert payload["layout"]["product"]["label"] == "Product"


def test_from_dataframe_missing_pandas(monkeypatch):
    import sys

    from isis_report_lib import inputs

    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pip install isis-report-lib\\[pandas\\]"):
        inputs.from_dataframe(object())


def test_from_csv_file_and_handle(tmp_path):
    from isis_report_lib.inputs import from_csv

    src = tmp_path / "in.csv"
    src.write_text("product,total,active\nPen,10.5,true\nPencil,2,\n", encoding="utf-8")
    payload = from_csv(src)
    assert payload["data"] == [
        {"product": "Pen", "total": 10.5, "active": True},
        {"product": "Pencil", "total": 2, "active": ""},
    ]
    with open(src, encoding="utf-8") as handle:
        assert from_csv(handle, meta={"title": "T"}, delimiter=",")["meta"] == {"title": "T"}
    semi = tmp_path / "semi.csv"
    semi.write_text("a;b\n1;2\n", encoding="utf-8")
    assert from_csv(semi, delimiter=";")["data"] == [{"a": 1, "b": 2}]


def test_from_json_list_dict_and_errors(tmp_path):
    from isis_report_lib.inputs import from_json

    src = tmp_path / "in.json"
    src.write_text('[{"a": 1}]', encoding="utf-8")
    assert from_json(src) == {"data": [{"a": 1}]}
    assert from_json(src, meta={"title": "T"})["meta"] == {"title": "T"}
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"meta": {"title": "A"}, "data": [{"a": 1}]}), encoding="utf-8")
    assert from_json(full)["meta"] == {"title": "A"}
    assert from_json(full, meta={"title": "B"})["meta"] == {"title": "B"}
    assert from_json([{"a": 1}]) == {"data": [{"a": 1}]}
    assert from_json({"meta": {"title": "A"}}, layout={"x": {}})["layout"] == {"x": {}}
    bad = tmp_path / "bad.json"
    bad.write_text("42", encoding="utf-8")
    with pytest.raises(ValueError, match="list or a payload object"):
        from_json(bad)


def test_payload_schema_shape():
    from isis_report_lib import payload_schema

    schema = payload_schema()
    assert schema["title"] == "ReportPayload"
    assert set(schema["$defs"]) >= {"MetaInfo"}
    assert "meta" in schema["properties"] and "data" in schema["properties"]


def test_clean_value_edge_cases():
    pd = pytest.importorskip("pandas")

    from isis_report_lib.inputs import _clean_value

    assert _clean_value([1, 2], pd) == [1, 2]  # pd.isna() unusable -> passthrough
    assert _clean_value("x", pd) == "x"


def test_from_json_file_handle():
    from isis_report_lib.inputs import from_json

    assert from_json(io.StringIO('[{"a": 1}]')) == {"data": [{"a": 1}]}


def test_from_jsonl_valid_file_path(tmp_path):
    from isis_report_lib.inputs import from_jsonl

    src = tmp_path / "data.jsonl"
    src.write_text('{"x": 10}\n{"y": 20}\n', encoding="utf-8")
    payload = from_jsonl(src)
    assert payload["data"] == [{"x": 10}, {"y": 20}]


def test_from_jsonl_valid_file_handle():
    from isis_report_lib.inputs import from_jsonl

    handle = io.StringIO('{"a": 1}\n{"b": 2}\n')
    payload = from_jsonl(handle)
    assert payload["data"] == [{"a": 1}, {"b": 2}]


def test_from_jsonl_skips_blank_lines():
    from isis_report_lib.inputs import from_jsonl

    content = '{"a": 1}\n\n{"b": 2}\n   \n{"c": 3}\n'
    payload = from_jsonl(io.StringIO(content))
    assert payload["data"] == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_from_jsonl_meta_merged(tmp_path):
    from isis_report_lib.inputs import from_jsonl

    src = tmp_path / "data.jsonl"
    src.write_text('{"x": 10}\n', encoding="utf-8")
    payload = from_jsonl(src, meta={"title": "T"})
    assert payload["meta"] == {"title": "T"}
    assert payload["data"] == [{"x": 10}]


def test_from_jsonl_layout_merged(tmp_path):
    from isis_report_lib.inputs import from_jsonl

    src = tmp_path / "data.jsonl"
    src.write_text('{"x": 10}\n', encoding="utf-8")
    payload = from_jsonl(src, layout={"x": {"label": "X"}})
    assert payload["layout"] == {"x": {"label": "X"}}
    assert payload["data"] == [{"x": 10}]


def test_from_jsonl_malformed_json_raises_with_line_number():
    from isis_report_lib.inputs import from_jsonl

    bad = io.StringIO('{"a": 1}\n{invalid}\n')
    with pytest.raises(ValueError, match="Malformed JSON on line 2"):
        from_jsonl(bad)


def test_from_jsonl_non_object_raises_with_line_number():
    from isis_report_lib.inputs import from_jsonl

    with pytest.raises(ValueError, match="JSON value on line 1 is not an object"):
        from_jsonl(io.StringIO("[1, 2]\n"))


def test_from_jsonl_does_not_close_caller_handle():
    from isis_report_lib.inputs import from_jsonl

    handle = io.StringIO('{"a": 1}\n')
    from_jsonl(handle)
    assert not handle.closed


def test_from_jsonl_excessive_nesting_raises_with_line_number():
    from isis_report_lib.inputs import from_jsonl

    # Build a deeply nested structure that exceeds Python's recursion limit
    deep = "[" * 1200 + "]" * 1200
    with pytest.raises(ValueError, match=r"line 1"):
        from_jsonl(io.StringIO(deep))
