"""
phase_a_tests -- self-test code for run_phase_a.py, extracted verbatim
(mechanical move only, no logic changes) from that module's former
"Self-test" section.

This package is only ever imported lazily, from inside
run_phase_a.py's own `main()` when `--self-test` is requested. It is
never imported by any real (--dry-run / --smoke-test / --official-run)
code path, and production code in run_phase_a.py never imports
anything from here.
"""
