from voice_loop import parse_control_prefix


def test_parse_control_prefix_defaults_to_speak_without_prefix():
    control, text = parse_control_prefix("hello there")
    assert control == "S"
    assert text == "hello there"


def test_parse_control_prefix_handles_passive_code():
    control, text = parse_control_prefix("\x1fP stay quiet")
    assert control == "P"
    assert text == "stay quiet"


def test_parse_control_prefix_handles_interject_code():
    control, text = parse_control_prefix("\x1fI breaking in")
    assert control == "I"
    assert text == "breaking in"


def test_parse_control_prefix_ignores_unknown_code():
    control, text = parse_control_prefix("\x1fZ fallback")
    assert control == "S"
    assert text == "\x1fZ fallback"
