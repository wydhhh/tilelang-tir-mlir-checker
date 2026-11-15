"""TileLang TIR -> structured MLIR inspection, without an embedded model."""

__version__ = "0.12.0"


def check(trace_dir, *, out_dir=None, dynamic=False, semantic=False, case=None, mlir_opt=None,
          seed=0, timeout=120, max_steps=2_000_000, stages=None, operator_name=None,
          translation=None, tir_backend='ast', tir_python=None,mlir_backend='text',ir_python=None):
    """Inspect one trace group; optionally write a report and evidence archive.

    Returns a JSON-serializable result. This does not assert semantic equivalence.
    Explicit case adapters run only inside an isolated subprocess. Supported IR
    subsets are interpreted independently, without invoking TIR->MLIR lowering.
    """
    from .analysis import analyze
    from .report import write_report
    from .naming import identify_operator, validate_operator_name

    if operator_name is not None:
        validate_operator_name(operator_name)

    result = analyze(trace_dir, dynamic=dynamic, semantic=semantic, case=case, mlir_opt=mlir_opt,
                     seed=seed, timeout=timeout, max_steps=max_steps, stages=stages,
                     translation=translation, tir_backend=tir_backend, tir_python=tir_python,
                     mlir_backend=mlir_backend,ir_python=ir_python)
    result['operator'] = identify_operator(result, operator_name)
    if out_dir is not None:
        write_report(result, out_dir)
    return result


__all__ = ["check", "__version__"]
