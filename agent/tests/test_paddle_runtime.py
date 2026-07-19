from agent.tools.paddle_ocr_runner import _validate_paddle_version


def test_paddle_runtime_accepts_declared_version():
    class FakePaddle:
        __version__ = "3.0.0"

    _validate_paddle_version(FakePaddle())


def test_paddle_runtime_rejects_undeclared_version():
    class FakePaddle:
        __version__ = "2.6.2"

    try:
        _validate_paddle_version(FakePaddle())
    except RuntimeError as exc:
        assert "installed=2.6.2, required=3.0.0" in str(exc)
    else:
        raise AssertionError("Undeclared PaddlePaddle runtime must fail")
