from get_names import count


class TestCount:
    def test_returns_expected_result(self):
        data = ["A", "A", "B", "A", "B"]
        expected = {"A": 3, "B": 2}
        result = count(data)
        assert result == expected
