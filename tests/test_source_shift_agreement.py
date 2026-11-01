from gates.act.source_shift_agreement import _select_context


def test_source_shift_module_exposes_locked_context_selector():
    assert callable(_select_context)
