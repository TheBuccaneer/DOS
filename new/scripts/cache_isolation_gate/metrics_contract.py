"""Attach the full EXISTING sampler history after stopping it, including failure paths."""
import copy


async def finalize_existing_sampler(episode_result, sampler, *, clock_domain):
    """Call in a finally block with an already allocated episode result.

    This records observations only. It does not start requests or choose cohorts.
    Existing trigger.metrics_raw_samples remains the historical trigger snapshot.
    """
    error = None
    try:
        if sampler is not None:
            await sampler.stop()
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        episode_result['metrics_full_samples'] = copy.deepcopy(sampler.samples if sampler is not None else [])
        episode_result['metrics_capture'] = {'clock_domain': copy.deepcopy(clock_domain),
                                            'sampler_available': sampler is not None,
                                            'stop_error': error,
                                            'coverage_validated': False}
    return episode_result
