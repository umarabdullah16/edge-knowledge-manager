from src import rag_processor


def test_safe_eval_math_expression_basic_arithmetic():
    assert rag_processor._safe_eval_math_expression("2 + 3 * 4") == 14.0


def test_safe_eval_math_expression_functions_and_constants():
    value = rag_processor._safe_eval_math_expression("sqrt(16) + pi")
    assert 7.14 < value < 7.15


def test_safe_eval_math_expression_rejects_unsafe_code():
    try:
        rag_processor._safe_eval_math_expression("__import__('os').system('echo hacked')")
        assert False, "Expected ValueError for unsafe code"
    except ValueError:
        assert True


def test_math_tool_context_formats_result():
    context = rag_processor._math_tool_context("calculate 10/4")
    assert "Expression:" in context
    assert "Result:" in context
    assert "2.5" in context
