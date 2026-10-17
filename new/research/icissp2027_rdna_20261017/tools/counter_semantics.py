"""Execute only two pure AST definitions from pinned source snapshot, never import vLLM.

These are synthetic scheduler outputs, not GPU or serving experiments.
"""
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / 'reference/vllm_v0_17_1/v1/utils.py'


def load_pure_counter():
    code = SOURCE.read_text()
    tree = ast.parse(code)
    keep = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef))
            and x.name in ('IterationDetails', 'compute_iteration_details')]
    assert len(keep) == 2
    env = {'dataclass': dataclasses.dataclass, 'SchedulerOutput': SimpleNamespace}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SOURCE), 'exec'), env)
    return env['compute_iteration_details']


def run_cases():
    counter = load_pure_counter()
    cases = []
    for name, context, new, count in [
        ('new_request_17_scheduled', False, True, 17),
        ('zero_output_request_17_scheduled', True, False, 17),
        ('positive_output_request_17_scheduled', False, False, 17),
        ('positive_output_request_1_scheduled', False, False, 1),
    ]:
        output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id='r')] if new else [],
            scheduled_cached_reqs=SimpleNamespace(is_context_phase=lambda _: context),
            num_scheduled_tokens={'r': count})
        result = dataclasses.asdict(counter(output))
        cases.append(dict(case=name, synthetic=True, scheduled_tokens=count, result=result))
    return cases


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    out = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               method='pure AST extraction; no vLLM import, CUDA, server, or network',
               cases=run_cases())
    (root / 'evidence/counter_semantics.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
