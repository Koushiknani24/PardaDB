"""Request validation helpers used by every API endpoint."""

import pytest

from pardadb import server


def test_int_param_accepts_values_in_range():
    assert server.int_param({"limit": "25"}, "limit", 10, 1, 100) == 25
    assert server.int_param({}, "limit", 10, 1, 100) == 10


@pytest.mark.parametrize("raw", ["abc", "1.5", "", None, "0", "101", "-3"])
def test_int_param_rejects_bad_or_out_of_range(raw):
    with pytest.raises(server.BadRequest):
        server.int_param({"limit": raw}, "limit", 10, 1, 100)


def test_str_param_limits_length_and_type():
    assert server.str_param({"user": "support_ravi"}, "user") == "support_ravi"
    assert server.str_param({}, "user") is None
    with pytest.raises(server.BadRequest):
        server.str_param({"user": "x" * 201}, "user")
    with pytest.raises(server.BadRequest):
        server.str_param({"user": ["a"]}, "user")
    with pytest.raises(server.BadRequest):
        server.str_param({}, "table", required=True)


def test_sql_param_requires_text_under_the_cap():
    assert server.sql_param({"sql": "  SELECT 1  "}) == "SELECT 1"
    with pytest.raises(server.BadRequest):
        server.sql_param({})
    with pytest.raises(server.BadRequest):
        server.sql_param({"sql": "SELECT " + "1," * server.MAX_SQL_CHARS})
