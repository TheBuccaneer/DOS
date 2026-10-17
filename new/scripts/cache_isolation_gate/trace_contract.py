"""Strict offline validation of the existing DOS Raw-SSE format."""
import re

IDENTITY_KEYS = ('prompt_sha256', 'prompt_token_ids_sent', 'generation_seed',
                 'expected_prompt_tokens', 'expected_completion_tokens')


def check_stream(record):
    if not isinstance(record, dict):
        return ['invalid_request']
    errors = []
    if record.get('status') != 'complete':
        errors.append('request_not_complete')
    events = record.get('raw_sse_events')
    if not isinstance(events, list) or not events:
        return errors + ['missing_raw_sse']
    start, end = record.get('request_start_ns'), record.get('stream_end_ns')
    bounds_ok = type(start) is int and type(end) is int and 0 <= start <= end
    if not bounds_ok:
        errors.append('request_time_bounds')
    tokens, token_times = [], []
    previous = None
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append('invalid_event')
            continue
        if type(event.get('event_index')) is not int or event['event_index'] != index:
            errors.append('event_index')
        ns, ids = event.get('receive_perf_counter_ns'), event.get('token_ids')
        if type(ns) is not int or ns < 0:
            errors.append('invalid_timestamp')
        else:
            if previous is not None and ns < previous:
                errors.append('nonmonotonic_timestamp')
            previous = ns
            if bounds_ok and not start <= ns <= end:
                errors.append('SSE_outside_request_bounds')
        if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
            errors.append('invalid_token_ids')
        else:
            tokens.extend(ids)
            if ids and type(ns) is int:
                token_times.append(ns)
    output = record.get('output_token_ids')
    if not isinstance(output, list) or output != tokens:
        errors.append('output_reconstruction')
    expected = record.get('expected_completion_tokens')
    if type(expected) is not int or expected < 1 or len(tokens) != expected:
        errors.append('completion_token_count')
    if not token_times:
        errors.append('no_output_tokens')
    else:
        if token_times[0] != record.get('first_token_receive_ns'):
            errors.append('first_token_timestamp')
        if token_times[-1] != record.get('last_token_receive_ns'):
            errors.append('last_token_timestamp')
    return sorted(set(errors))


def request_identities(episode):
    rows = episode.get('victim_requests')
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty_or_missing_decode_population')
    identities = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid_request')
        index = row.get('request_index')
        if type(index) is not int or index < 0 or index in identities:
            raise ValueError('invalid_or_duplicate_request_index')
        if any(key not in row or row[key] is None for key in IDENTITY_KEYS):
            raise ValueError('missing_request_identity_fields')
        if not isinstance(row['prompt_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['prompt_sha256']):
            raise ValueError('invalid_prompt_sha256')
        ids = row['prompt_token_ids_sent']
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('invalid_prompt_tokens')
        if type(row['generation_seed']) is not int or row['generation_seed'] < 0:
            raise ValueError('invalid_generation_seed')
        if type(row['expected_prompt_tokens']) is not int or row['expected_prompt_tokens'] != len(ids):
            raise ValueError('prompt_token_count_mismatch')
        if type(row['expected_completion_tokens']) is not int or row['expected_completion_tokens'] < 1:
            raise ValueError('invalid_expected_completion_tokens')
        identities[index] = {key: row[key] for key in IDENTITY_KEYS}
    return identities
