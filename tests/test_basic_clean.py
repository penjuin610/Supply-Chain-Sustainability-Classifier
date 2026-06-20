from src.cleaning.basic_clean import basic_clean


def test_basic_clean_removes_known_noise() -> None:
    text = "Page 31 of 67 CONFIDENTIAL Sustainability accounts for 15% of score."
    cleaned = basic_clean(text)
    assert "Page 31 of 67" not in cleaned
    assert "CONFIDENTIAL" not in cleaned
    assert "15%" in cleaned
    assert "Sustainability" in cleaned
