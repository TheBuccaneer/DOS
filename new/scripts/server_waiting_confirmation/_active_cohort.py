"""
_active_cohort.py -- data-driven active-cohort detection and trigger
barrier for the server-waiting confirmation runner.

This module is the direct replacement for the client-semaphore-based
"wave" logic (_compute_wave / static request_index < concurrency) used
by the audited Prefill-Confirmation runner. Here, the "active cohort"
(the K requests actually running on vLLM's server-side scheduler,
where K = server_max_num_seqs) is never assumed from request_index. It
is determined ONLY from the observed order of first-token arrivals.

Two-phase protocol per episode:

  Phase 1 -- cohort freeze:
    All 20 victim requests are dispatched with no client admission
    semaphore. Each request's first output-token receive timestamp
    (highest-resolution monotonic clock available) is recorded. Once at
    least K distinct requests have received a first token, the cohort
    is determined by ranking ALL currently-observed first-token events
    by (timestamp, request_index) and taking the earliest K. If the
    K-th and (K+1)-th ranked timestamps are equal (indistinguishable at
    the recorded clock resolution), the cohort is AMBIGUOUS and the
    episode must be invalidated -- never resolved arbitrarily.

    A provisional "OK" determination is DEBOUNCED: it is only accepted
    once the same (active_indices, cohort_freeze_ns) pair is observed
    again after at least one full event-loop yield. This closes a
    scheduling race described below (see "cohort-freeze debounce").

  Phase 2 -- logical trigger:
    Once the cohort is frozen unambiguously, a threshold-crossing
    barrier (reusing the audited _ActiveWaveCrossing /
    make_threshold_callback pattern from run_prefill_confirmation.py,
    which is already generic over an arbitrary index set) is armed for
    exactly the frozen cohort members, watching for each to reach
    `trigger_after_decode_tokens` cumulative output tokens. The logical
    trigger timestamp is the maximum of the per-member crossing
    timestamps. At that instant, every victim OUTSIDE the cohort must
    still have zero output tokens; otherwise the episode is invalid.

No GPU, network, or vLLM dependency. Pure asyncio + dataclasses.

=============================================================================
Revision note (this version)
=============================================================================
This revision fixes four issues identified in review of the previous
version of this module:

1. No more fixed 50 ms polling. Both phases now block on a shared,
   generation-counted `_ProgressSignal` that is bumped every time any
   request records new progress, and wake up immediately on that
   signal rather than on a periodic timeout. The timeout arguments now
   mean exactly what they say -- an overall deadline -- not a polling
   granularity.

2. Cohort-freeze debounce (closes the K/K+1 scheduling race). Reaching
   >= K observed first-token events in a single snapshot is NOT
   sufficient to freeze the cohort: a sibling request may already have
   delivered its first token at the OS/transport level but not yet had
   its coroutine scheduled to record that fact into `progress_by_index`
   (cooperative scheduling only guarantees one coroutine runs at a
   time between await points -- it says nothing about the arrival
   order of "ready" callbacks across independent request tasks). A
   provisional OK cohort is therefore re-confirmed after at least one
   `await asyncio.sleep(0)` yield; if the recomputed
   (active_indices, cohort_freeze_ns) differs, the provisional result
   is discarded and Phase 1 continues with the updated data. This is a
   bounded mitigation, not a proof of physical simultaneity -- true
   tie detection at the recorded clock resolution is still handled by
   `determine_active_cohort`'s AMBIGUOUS path.

3. No orphaned wait-tasks. Every `asyncio.ensure_future(event.wait())`
   wrapper created to multiplex a wait is explicitly cancelled and
   awaited (with `return_exceptions=True`) as soon as the surrounding
   `asyncio.wait(...)` call returns, in all cases (timeout, first-
   completed, or cohort confirmed). No pending wrapper tasks are left
   for the garbage collector across iterations of a long run.

4. Completion is checked by timestamp, not by `task.done()`. Each
   `TokenProgress` now carries an explicit `completion_ns`, set only
   when the caller marks a record as final (`record(..., is_final=True)`
   or `mark_complete(ns)`). Phase 2 checks `completion_ns` against the
   logical trigger directly ("no active request completed before
   trigger", i.e. `completion_ns` must not be <= `logical_trigger_ns`)
   instead of inferring completion from `task.done()`, which cannot
   establish *when* relative to the trigger a request finished --
   including the edge case where a single streaming chunk carries a
   cohort member directly from below-threshold to its final token.

   IMPORTANT SCOPE NOTE: this module can only check `completion_ns`
   for requests that have already completed by the time Phase 2
   returns (i.e. by the time the trigger condition is satisfied).
   A cohort member is free to complete strictly AFTER this function
   returns (in fact this is the expected common case, since output
   length 64 > trigger threshold 16). Confirming, for every cohort
   member, that its eventual completion timestamp is not <= the
   logical trigger is therefore also -- and primarily -- the
   responsibility of the full post-episode validator in the runner,
   which has access to every request's final `completion_ns` after
   all 20 victims have actually finished. That validator does not
   exist in this file (see the accompanying message for why).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable


# ============================================================================
# Shared wake-up signal: generation-counted to avoid the classic
# "lost wakeup" race between an Event.clear() and a concurrent Event.set().
# ============================================================================

class _ProgressSignal:
    """A wake-up signal safe to use from a single-threaded event loop.

    A plain `asyncio.Event` is unsafe to reuse across a check/clear/wait
    loop: if `set()` happens between the waiter's last check and its
    `clear()`, that wakeup is silently lost. This wrapper instead keeps
    a monotonically increasing generation counter; a waiter records the
    generation it last observed and is only allowed to block if the
    generation has not moved since, closing that race.
    """

    __slots__ = ("_event", "generation")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.generation = 0

    def bump(self) -> None:
        self.generation += 1
        self._event.set()

    async def wait_for_change(self, since_generation: int, timeout: float | None) -> int:
        """Return as soon as `generation != since_generation`, either
        immediately (if already true) or after being woken by `bump()`,
        or after `timeout` seconds, whichever comes first. Always
        returns the generation observed at return time. Cleans up any
        wrapper task it creates before returning, in every case.
        """
        if self.generation != since_generation:
            return self.generation
        self._event.clear()
        if self.generation != since_generation:
            # Defensive re-check: on a single-threaded loop this branch
            # cannot be reached between the two checks above (no await
            # in between), but is kept in case this method is ever
            # reused in a context where that invariant no longer holds.
            return self.generation

        waiter = asyncio.ensure_future(self._event.wait())
        try:
            await asyncio.wait([waiter], timeout=timeout)
        finally:
            if not waiter.done():
                waiter.cancel()
            try:
                await waiter
            except (asyncio.CancelledError, Exception):
                pass
        return self.generation


async def _yield_once() -> None:
    """A single full event-loop tick, used for the cohort-freeze
    debounce (issue 2) -- gives any coroutine that is already runnable
    but has not yet had its turn a chance to apply its update."""
    await asyncio.sleep(0)


# ============================================================================
# Phase 1: pure, directly-testable cohort determination
# ============================================================================

COHORT_STATUS_OK = "ok"
COHORT_STATUS_INSUFFICIENT_DATA = "insufficient_data"
COHORT_STATUS_AMBIGUOUS = "ambiguous_active_cohort_boundary"


@dataclass
class CohortResult:
    status: str
    active_indices: frozenset[int] | None
    cohort_freeze_ns: int | None
    ordering: list[tuple[int, int]]  # (request_index, first_token_perf_ns), sorted
    detail: str


def determine_active_cohort(
    first_token_ns_by_index: dict[int, int | None],
    k: int,
) -> CohortResult:
    """
    Ranks every request that has (so far) received a first output token
    by (first_token_perf_ns, request_index) -- the request_index is used
    strictly as a tie-break for otherwise-distinguishable timestamps,
    never to resolve a true cross-boundary simultaneity (see module
    docstring and requirement 5/6). Returns:

      - insufficient_data  if fewer than k requests have a first token yet
                           (caller should keep waiting),
      - ambiguous_active_cohort_boundary
                           if the k-th and (k+1)-th ranked timestamps are
                           equal -- the boundary cannot be established at
                           the available clock resolution,
      - ok                 with the frozen set of exactly k active indices
                           and cohort_freeze_ns = the k-th ranked
                           timestamp (the instant the cohort became
                           knowable).

    This function is pure and does not itself resolve the K/K+1
    scheduling race described in the module docstring (issue 2) --
    that is the caller's responsibility (see
    `watch_dynamic_cohort_and_trigger`'s debounce loop), because this
    function only ever sees a single snapshot and has no notion of
    "wait for more data".
    """
    if type(k) is not int or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")

    observed = [
        (idx, ns) for idx, ns in first_token_ns_by_index.items() if type(ns) is int
    ]
    if len(observed) < k:
        return CohortResult(
            status=COHORT_STATUS_INSUFFICIENT_DATA,
            active_indices=None,
            cohort_freeze_ns=None,
            ordering=sorted(observed, key=lambda t: (t[1], t[0])),
            detail=f"only {len(observed)} of required {k} first-token events observed",
        )

    ordered = sorted(observed, key=lambda t: (t[1], t[0]))
    kth_ns = ordered[k - 1][1]

    if len(ordered) > k and ordered[k][1] == kth_ns:
        return CohortResult(
            status=COHORT_STATUS_AMBIGUOUS,
            active_indices=None,
            cohort_freeze_ns=kth_ns,
            ordering=ordered,
            detail=(
                f"the {k}-th and {k + 1}-th first-token timestamps are both "
                f"{kth_ns} ns; cannot establish a strict temporal order at "
                f"the recorded clock resolution"
            ),
        )

    active = frozenset(idx for idx, _ns in ordered[:k])
    return CohortResult(
        status=COHORT_STATUS_OK,
        active_indices=active,
        cohort_freeze_ns=kth_ns,
        ordering=ordered,
        detail="",
    )


# ============================================================================
# Phase 2: threshold-crossing barrier for a (now-known) index set -- this is
# intentionally the same shape as the audited _ActiveWaveCrossing /
# make_threshold_callback pattern in run_prefill_confirmation.py, which was
# already generic over an arbitrary set of indices; only the CALLER now
# supplies a data-driven set instead of a static request_index < concurrency
# wave.
# ============================================================================

@dataclass
class _ActiveCohortCrossing:
    request_index: int
    threshold_crossing_ns: int | None = None
    received_token_count_at_crossing: int | None = None
    token_count_history: list[tuple[int, int]] = field(default_factory=list)


def make_cohort_threshold_callback(
    request_index: int,
    threshold: int,
    crossing: _ActiveCohortCrossing,
    event: asyncio.Event,
    signal: "_ProgressSignal | None" = None,
) -> Callable[[int, int], None]:
    """Identical semantics to make_threshold_callback() in
    run_prefill_confirmation.py: fires exactly once, at the first token
    BATCH whose cumulative count reaches or exceeds `threshold`.

    If `signal` is supplied (the shared `_ProgressSignal` used by
    `watch_dynamic_cohort_and_trigger`), it is bumped on every call so
    that a real (non-test) caller wiring this callback directly into
    request streaming code wakes the watcher immediately -- consistent
    with `TokenProgress.record`'s own bump-on-update behaviour.
    """

    def _cb(cumulative_count: int, receive_ns: int) -> None:
        crossing.token_count_history.append((receive_ns, cumulative_count))
        if crossing.threshold_crossing_ns is None and cumulative_count >= threshold:
            crossing.threshold_crossing_ns = receive_ns
            crossing.received_token_count_at_crossing = cumulative_count
            if not event.is_set():
                event.set()
        if signal is not None:
            signal.bump()

    return _cb


@dataclass
class TokenProgress:
    """Cumulative output-token progress for ONE victim request, tracked
    for ALL 20 requests (not just the cohort) so that the "zero tokens
    outside the cohort at the logical trigger" invariant (requirement
    10) can be checked against real data rather than assumed.

    `signal`, if supplied, is bumped on every `record()` call so that
    `watch_dynamic_cohort_and_trigger` wakes immediately instead of
    polling (issue 1). `completion_ns` is set only when a record is
    explicitly marked final, and is what Phase 2 uses to check "no
    active request completed before trigger" by timestamp rather than
    by `task.done()` (issue 4).
    """

    request_index: int
    signal: "_ProgressSignal | None" = None
    first_token_ns: int | None = None
    cumulative_count: int = 0
    completion_ns: int | None = None
    history: list[tuple[int, int]] = field(default_factory=list)  # (ns, cumulative_count)

    def record(self, cumulative_count: int, receive_ns: int, *, is_final: bool = False) -> None:
        if self.first_token_ns is None and cumulative_count > 0:
            self.first_token_ns = receive_ns
        self.cumulative_count = cumulative_count
        self.history.append((receive_ns, cumulative_count))
        if is_final:
            self.completion_ns = receive_ns
        if self.signal is not None:
            self.signal.bump()

    def mark_complete(self, receive_ns: int) -> None:
        """Explicit completion marker for callers whose final chunk is
        a distinct event from the last token record (e.g. a trailing
        stream-close bookkeeping step)."""
        self.completion_ns = receive_ns
        if self.signal is not None:
            self.signal.bump()

    def count_at_or_before(self, ns: int) -> int:
        """Cumulative token count as of the latest recorded sample at or
        before `ns` (0 if no sample yet at/before that time)."""
        count = 0
        for sample_ns, sample_count in self.history:
            if sample_ns <= ns:
                count = sample_count
            else:
                break
        return count


DYNAMIC_TRIGGER_OK = "ok"
DYNAMIC_TRIGGER_TIMEOUT = "timeout"
DYNAMIC_TRIGGER_PRETRIGGER_FAILURE = "pretrigger_failure"
DYNAMIC_TRIGGER_AMBIGUOUS_COHORT = "ambiguous_active_cohort_boundary"
DYNAMIC_TRIGGER_COHORT_TIMEOUT = "cohort_freeze_timeout"
DYNAMIC_TRIGGER_NONCOHORT_LEAK = "noncohort_token_before_trigger"


@dataclass
class DynamicTriggerResult:
    status: str
    active_indices: frozenset[int] | None
    cohort_freeze_ns: int | None
    logical_trigger_ns: int | None
    detail: str


async def watch_dynamic_cohort_and_trigger(
    *,
    k: int,
    trigger_after_decode_tokens: int,
    first_token_events: dict[int, asyncio.Event],
    progress_by_index: dict[int, TokenProgress],
    victim_tasks: dict[int, "asyncio.Task"],
    request_status_ok: Callable[[dict], bool],
    cohort_freeze_timeout_s: float,
    trigger_timeout_s: float,
    signal: "_ProgressSignal | None" = None,
) -> DynamicTriggerResult:
    """
    Orchestrates both phases against already-running victim tasks (the
    caller is responsible for having created and dispatched all 20
    tasks with no client admission semaphore before calling this).

    `first_token_events[i]` must be set by the request-i coroutine the
    moment it receives its first output token (any token, not the
    trigger_after_decode_tokens-th) -- this remains a documented part
    of the external contract (some callers use it for first-token
    latency bookkeeping outside this module), but this function no
    longer waits on these events individually; it waits on `signal`.

    `progress_by_index[i]` must be updated (via TokenProgress.record)
    on every received token batch for every one of the 20 requests,
    including non-cohort ones, so the post-freeze invariant check has
    real data. Every `TokenProgress` in `progress_by_index` SHOULD
    share the same `_ProgressSignal` instance passed as `signal` here
    (construct it once per episode and hand it to both), otherwise
    this function falls back to the deadline as its only wakeup and
    behaves as if no progress signal were wired -- callers are
    responsible for this wiring; it cannot be enforced from here.
    """
    own_signal = signal if signal is not None else _ProgressSignal()

    # --- Phase 1: wait until >= k requests have a first token, rank them,
    # and DEBOUNCE the resulting cohort across one event-loop yield before
    # accepting it (issue 2).
    start = time.monotonic()
    last_seen_generation = own_signal.generation
    provisional: tuple[frozenset[int], int] | None = None

    while True:
        observed = {
            idx: progress.first_token_ns
            for idx, progress in progress_by_index.items()
            if progress.first_token_ns is not None
        }
        cohort = determine_active_cohort(observed, k)

        if cohort.status == COHORT_STATUS_AMBIGUOUS:
            return DynamicTriggerResult(
                status=DYNAMIC_TRIGGER_AMBIGUOUS_COHORT,
                active_indices=None,
                cohort_freeze_ns=cohort.cohort_freeze_ns,
                logical_trigger_ns=None,
                detail=cohort.detail,
            )

        if cohort.status == COHORT_STATUS_OK:
            candidate = (cohort.active_indices, cohort.cohort_freeze_ns)
            if provisional == candidate:
                # Confirmed stable across at least one full event-loop
                # yield: no sibling coroutine that was already runnable
                # changed the outcome. Freeze for real.
                break
            provisional = candidate
            await _yield_once()
            continue

        # insufficient_data: block on the shared signal (event-driven,
        # not a fixed poll interval) until either more progress arrives
        # or the overall deadline is hit.
        provisional = None
        remaining = cohort_freeze_timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            return DynamicTriggerResult(
                status=DYNAMIC_TRIGGER_COHORT_TIMEOUT,
                active_indices=None,
                cohort_freeze_ns=None,
                logical_trigger_ns=None,
                detail=f"fewer than {k} first-token events observed within {cohort_freeze_timeout_s}s",
            )
        last_seen_generation = await own_signal.wait_for_change(last_seen_generation, timeout=remaining)

    active_indices = cohort.active_indices
    cohort_freeze_ns = cohort.cohort_freeze_ns

    # --- Invariant (requirement 7): at freeze, nothing outside the cohort
    # already has an earlier first-token timestamp than any cohort member.
    # The debounce loop above already makes this violation unlikely to
    # slip through undetected (a late-but-earlier sibling would normally
    # have changed `candidate` on the confirming pass and looped back into
    # Phase 1 instead of reaching here) -- this is kept as a defensive
    # re-check against the live progress data for any residual scheduling
    # anomaly the debounce window did not happen to catch.
    for idx, progress in progress_by_index.items():
        if idx in active_indices:
            continue
        if progress.first_token_ns is not None and progress.first_token_ns <= cohort_freeze_ns:
            return DynamicTriggerResult(
                status=DYNAMIC_TRIGGER_NONCOHORT_LEAK,
                active_indices=active_indices,
                cohort_freeze_ns=cohort_freeze_ns,
                logical_trigger_ns=None,
                detail=(
                    f"request {idx} outside the frozen cohort already had a "
                    f"first token at or before cohort_freeze_ns={cohort_freeze_ns} "
                    f"(survived the cohort-freeze debounce -- treat as a data "
                    f"quality anomaly, not a normal outcome)"
                ),
            )

    # --- Phase 2: threshold-crossing barrier for exactly the frozen cohort.
    # crossing_ns is derived from progress_by_index histories (kept
    # self-contained, independent of request-execution plumbing); a real
    # runner wiring make_cohort_threshold_callback directly will bump
    # `own_signal` itself and this derivation stays correct either way.
    def _check_thresholds_from_progress() -> dict[int, int | None]:
        crossing_ns: dict[int, int | None] = {}
        for idx in active_indices:
            progress = progress_by_index[idx]
            ns_at_cross = None
            for sample_ns, sample_count in progress.history:
                if sample_count >= trigger_after_decode_tokens:
                    ns_at_cross = sample_ns
                    break
            crossing_ns[idx] = ns_at_cross
        return crossing_ns

    trigger_start = time.monotonic()
    while True:
        crossing_ns = _check_thresholds_from_progress()
        if all(ns is not None for ns in crossing_ns.values()):
            logical_trigger_ns = max(crossing_ns.values())

            # Requirement 10, at the logical trigger instant.
            for idx, progress in progress_by_index.items():
                if idx in active_indices:
                    continue
                if progress.count_at_or_before(logical_trigger_ns) > 0:
                    return DynamicTriggerResult(
                        status=DYNAMIC_TRIGGER_NONCOHORT_LEAK,
                        active_indices=active_indices,
                        cohort_freeze_ns=cohort_freeze_ns,
                        logical_trigger_ns=logical_trigger_ns,
                        detail=(
                            f"request {idx} outside the frozen cohort already "
                            f"emitted output token(s) at or before the logical "
                            f"trigger ({logical_trigger_ns} ns)"
                        ),
                    )

            # Issue 4: timestamp-based completion check, not task.done().
            # Only catches cohort members that HAVE ALREADY completed by
            # the time the trigger fires (see module docstring scope
            # note) -- members that complete later must be checked by the
            # runner's post-episode validator using the returned
            # logical_trigger_ns and each TokenProgress.completion_ns.
            for idx in active_indices:
                completion_ns = progress_by_index[idx].completion_ns
                if completion_ns is not None and completion_ns <= logical_trigger_ns:
                    return DynamicTriggerResult(
                        status=DYNAMIC_TRIGGER_PRETRIGGER_FAILURE,
                        active_indices=active_indices,
                        cohort_freeze_ns=cohort_freeze_ns,
                        logical_trigger_ns=logical_trigger_ns,
                        detail=(
                            f"cohort request {idx} has completion_ns="
                            f"{completion_ns} <= logical_trigger_ns="
                            f"{logical_trigger_ns} (must be a request-index bug, "
                            f"a mis-timed final chunk, or output length < "
                            f"trigger_after_decode_tokens)"
                        ),
                    )

            for idx in active_indices:
                task = victim_tasks[idx]
                if task.done():
                    try:
                        result = task.result()
                    except BaseException:
                        return DynamicTriggerResult(
                            status=DYNAMIC_TRIGGER_PRETRIGGER_FAILURE,
                            active_indices=active_indices,
                            cohort_freeze_ns=cohort_freeze_ns,
                            logical_trigger_ns=logical_trigger_ns,
                            detail=f"cohort request {idx} raised before the trigger could be confirmed",
                        )
                    if not request_status_ok(result):
                        return DynamicTriggerResult(
                            status=DYNAMIC_TRIGGER_PRETRIGGER_FAILURE,
                            active_indices=active_indices,
                            cohort_freeze_ns=cohort_freeze_ns,
                            logical_trigger_ns=logical_trigger_ns,
                            detail=f"cohort request {idx} completed with a non-ok status before the trigger",
                        )

            return DynamicTriggerResult(
                status=DYNAMIC_TRIGGER_OK,
                active_indices=active_indices,
                cohort_freeze_ns=cohort_freeze_ns,
                logical_trigger_ns=logical_trigger_ns,
                detail="",
            )

        for idx in active_indices:
            task = victim_tasks[idx]
            if task.done() and crossing_ns.get(idx) is None:
                return DynamicTriggerResult(
                    status=DYNAMIC_TRIGGER_PRETRIGGER_FAILURE,
                    active_indices=active_indices,
                    cohort_freeze_ns=cohort_freeze_ns,
                    logical_trigger_ns=None,
                    detail=f"cohort request {idx} finished before reaching its {trigger_after_decode_tokens}-token threshold",
                )

        remaining = trigger_timeout_s - (time.monotonic() - trigger_start)
        if remaining <= 0:
            return DynamicTriggerResult(
                status=DYNAMIC_TRIGGER_TIMEOUT,
                active_indices=active_indices,
                cohort_freeze_ns=cohort_freeze_ns,
                logical_trigger_ns=None,
                detail=f"cohort did not reach the {trigger_after_decode_tokens}-token barrier within {trigger_timeout_s}s",
            )

        pending_tasks = [victim_tasks[i] for i in active_indices if not victim_tasks[i].done()]
        if not pending_tasks:
            # All cohort tasks already done but we haven't returned above
            # (e.g. threshold reached exactly at completion) -- avoid a
            # busy loop, let the next iteration re-check.
            await asyncio.sleep(0)
            continue

        signal_waiter = asyncio.ensure_future(
            own_signal.wait_for_change(own_signal.generation, timeout=None)
        )
        task_waiters = list(pending_tasks)
        try:
            await asyncio.wait(
                [signal_waiter, *task_waiters],
                timeout=min(remaining, trigger_timeout_s),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not signal_waiter.done():
                signal_waiter.cancel()
                try:
                    await signal_waiter
                except (asyncio.CancelledError, Exception):
                    pass
            # task_waiters are the caller-owned victim_tasks themselves,
            # never cancelled or awaited here -- only the signal_waiter
            # wrapper this function created is ours to clean up (issue 3).


# ============================================================================
# Self-test: pure-Python, no network/GPU/asyncio-real-time dependency.
# Covers offline-test items 6, 7, 9, 10, 11, 12 from the implementation spec,
# plus two regression tests added in this revision (no-poll-delay,
# cohort-freeze debounce / K+1 scheduling race).
# ============================================================================

def _selftest_phase1() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    timestamps = {0: 1050, 1: 1010, 2: 1090, 3: 1020, 4: 1030, 5: 1200, 6: 1300}
    res = determine_active_cohort(timestamps, k=4)
    check("phase1: K=4 status ok", res.status == COHORT_STATUS_OK, res.detail)
    check(
        "phase1: K=4 cohort is data-driven (earliest 4 by timestamp), not request_index<4",
        res.active_indices == frozenset({1, 3, 4, 0}),
        str(res.active_indices),
    )
    check("phase1: cohort_freeze_ns is the K-th ranked timestamp", res.cohort_freeze_ns == 1050)

    ts8 = {i: 2000 + i for i in range(20)}
    res8 = determine_active_cohort(ts8, k=8)
    check("phase1: K=8 status ok", res8.status == COHORT_STATUS_OK)
    check("phase1: K=8 cohort has exactly 8 members", res8.active_indices is not None and len(res8.active_indices) == 8)

    res_a = determine_active_cohort(timestamps, k=4)
    res_b = determine_active_cohort(dict(timestamps), k=4)
    check(
        "phase1: deterministic ordering across repeated calls with identical input",
        res_a.active_indices == res_b.active_indices and res_a.cohort_freeze_ns == res_b.cohort_freeze_ns,
    )

    ambiguous_ts = {0: 100, 1: 110, 2: 120, 3: 130, 4: 130, 5: 999}
    res_amb = determine_active_cohort(ambiguous_ts, k=4)
    check(
        "phase1: equal K-th/(K+1)-th timestamps -> ambiguous_active_cohort_boundary",
        res_amb.status == COHORT_STATUS_AMBIGUOUS,
        res_amb.detail,
    )
    check("phase1: ambiguous result never carries an arbitrarily-chosen cohort", res_amb.active_indices is None)

    res_insuff = determine_active_cohort({0: 100, 1: 110}, k=4)
    check("phase1: fewer than K observed -> insufficient_data", res_insuff.status == COHORT_STATUS_INSUFFICIENT_DATA)

    return checks


def _selftest_phase2() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    def request_status_ok(result: dict) -> bool:
        return result.get("status") == "complete"

    async def happy_path():
        n, k, trigger_after = 8, 4, 16
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        cohort_first = {1: 1010, 3: 1020, 4: 1030, 0: 1050}
        cohort_cross = {1: 5000, 3: 5100, 4: 5300, 0: 5900}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def cohort_task(idx: int) -> dict:
            progress[idx].record(1, cohort_first[idx])
            first_token_events[idx].set()
            await asyncio.sleep(0)
            progress[idx].record(trigger_after, cohort_cross[idx])
            await asyncio.sleep(0)
            # Completion must be realistic: ALL cohort members request 64
            # output tokens and are dispatched at roughly the same time,
            # so no member finishes before the SLOWEST member's 16-token
            # crossing (the global trigger = max over the cohort), even
            # though each finishes comfortably after its OWN crossing.
            # Using max(cohort_cross.values()) here (not cohort_cross[idx])
            # is what makes this a realistic "happy path" fixture.
            progress[idx].record(trigger_after + 4, max(cohort_cross.values()) + 500, is_final=True)
            return {"status": "complete"}

        async def noncohort_task(idx: int) -> dict:
            await asyncio.sleep(0)
            progress[idx].record(1, 9000 + idx)
            first_token_events[idx].set()
            return {"status": "complete"}

        for i in (1, 3, 4, 0):
            victim_tasks[i] = asyncio.ensure_future(cohort_task(i))
        for i in (2, 5, 6, 7):
            victim_tasks[i] = asyncio.ensure_future(noncohort_task(i))

        result = await watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=trigger_after,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=5.0, trigger_timeout_s=5.0, signal=sig,
        )
        await asyncio.gather(*victim_tasks.values())
        return result

    async def noncohort_leak():
        n, k, trigger_after = 8, 4, 16
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        cohort_first = {1: 1010, 3: 1020, 4: 1030, 0: 1050}
        cohort_cross = {1: 5000, 3: 5100, 4: 5300, 0: 5900}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def cohort_task(idx: int) -> dict:
            progress[idx].record(1, cohort_first[idx])
            first_token_events[idx].set()
            await asyncio.sleep(0)
            progress[idx].record(trigger_after, cohort_cross[idx])
            return {"status": "complete"}

        async def leaking_task(idx: int) -> dict:
            await asyncio.sleep(0)
            progress[idx].record(1, 3000)  # well before the logical trigger (5900)
            first_token_events[idx].set()
            return {"status": "complete"}

        async def normal_task(idx: int) -> dict:
            await asyncio.sleep(0)
            progress[idx].record(1, 9000 + idx)
            first_token_events[idx].set()
            return {"status": "complete"}

        for i in (1, 3, 4, 0):
            victim_tasks[i] = asyncio.ensure_future(cohort_task(i))
        victim_tasks[2] = asyncio.ensure_future(leaking_task(2))
        for i in (5, 6, 7):
            victim_tasks[i] = asyncio.ensure_future(normal_task(i))

        result = await watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=trigger_after,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=5.0, trigger_timeout_s=5.0, signal=sig,
        )
        await asyncio.gather(*victim_tasks.values())
        return result

    async def cohort_freeze_timeout_case():
        n, k = 4, 4
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def stuck_task(idx: int) -> dict:
            await asyncio.sleep(10)
            return {"status": "complete"}

        async def token_task(idx: int) -> dict:
            progress[idx].record(1, 100 + idx)
            first_token_events[idx].set()
            await asyncio.sleep(10)
            return {"status": "complete"}

        victim_tasks[0] = asyncio.ensure_future(token_task(0))
        victim_tasks[1] = asyncio.ensure_future(token_task(1))
        victim_tasks[2] = asyncio.ensure_future(stuck_task(2))
        victim_tasks[3] = asyncio.ensure_future(stuck_task(3))

        result = await watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=16,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=0.2, trigger_timeout_s=0.2, signal=sig,
        )
        for t in victim_tasks.values():
            t.cancel()
        await asyncio.gather(*victim_tasks.values(), return_exceptions=True)
        return result

    async def completed_before_trigger_case():
        # Regression for issue 4: a cohort member whose single final chunk
        # crosses the 16-token threshold AND completes the request at the
        # same instant, while a slower cohort sibling pushes the logical
        # trigger (max of crossings) to a LATER timestamp. The fast
        # member's completion_ns then sits strictly before the logical
        # trigger -- must be caught by timestamp, which task.done() alone
        # cannot do (task.done() only says "finished", not "finished
        # before which trigger timestamp").
        n, k, trigger_after = 4, 2, 16
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def fast_jump_task(idx: int) -> dict:
            # First token AND final (64) token delivered in one jump.
            progress[idx].record(1, 100)
            first_token_events[idx].set()
            await asyncio.sleep(0)
            progress[idx].record(64, 5000, is_final=True)  # crosses 16 at ns=5000, completes at ns=5000
            return {"status": "complete"}

        async def slow_task(idx: int) -> dict:
            progress[idx].record(1, 110)
            first_token_events[idx].set()
            await asyncio.sleep(0)
            progress[idx].record(trigger_after, 9000)  # crosses 16 much later -> logical_trigger_ns = 9000
            await asyncio.sleep(10)
            return {"status": "complete"}

        async def idle_task(idx: int) -> dict:
            await asyncio.sleep(10)
            return {"status": "complete"}

        victim_tasks[0] = asyncio.ensure_future(fast_jump_task(0))
        victim_tasks[1] = asyncio.ensure_future(slow_task(1))
        victim_tasks[2] = asyncio.ensure_future(idle_task(2))
        victim_tasks[3] = asyncio.ensure_future(idle_task(3))

        result = await watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=trigger_after,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=2.0, trigger_timeout_s=2.0, signal=sig,
        )
        for t in victim_tasks.values():
            if not t.done():
                t.cancel()
        await asyncio.gather(*victim_tasks.values(), return_exceptions=True)
        return result

    async def no_poll_delay_case():
        # Regression for issue 1: the watcher must wake up promptly when
        # progress is recorded, not only after a ~50ms polling slice.
        # We record the K-th first-token event from a background task
        # after a short real-time delay and assert the watcher's own
        # wait returns quickly afterwards, not ~50ms+ later.
        n, k = 4, 2
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def delayed_task(idx: int, delay: float, ns: int) -> dict:
            await asyncio.sleep(delay)
            progress[idx].record(1, ns)
            first_token_events[idx].set()
            await asyncio.sleep(10)
            return {"status": "complete"}

        victim_tasks[0] = asyncio.ensure_future(delayed_task(0, 0.005, 1000))
        victim_tasks[1] = asyncio.ensure_future(delayed_task(1, 0.010, 2000))
        victim_tasks[2] = asyncio.ensure_future(delayed_task(2, 10.0, 3000))
        victim_tasks[3] = asyncio.ensure_future(delayed_task(3, 10.0, 4000))

        t0 = time.monotonic()
        watcher = asyncio.ensure_future(
            watch_dynamic_cohort_and_trigger(
                k=k, trigger_after_decode_tokens=1,
                first_token_events=first_token_events, progress_by_index=progress,
                victim_tasks=victim_tasks, request_status_ok=request_status_ok,
                cohort_freeze_timeout_s=5.0, trigger_timeout_s=5.0, signal=sig,
            )
        )
        result = await watcher
        elapsed = time.monotonic() - t0
        for t in victim_tasks.values():
            if not t.done():
                t.cancel()
        await asyncio.gather(*victim_tasks.values(), return_exceptions=True)
        return result, elapsed

    async def cohort_freeze_race_case():
        # Regression for issue 2: requests 0-3 record a first token with
        # NO await before doing so (they run to completion of that line
        # in the same event-loop pass their tasks are first scheduled).
        # Request 4 yields once (await asyncio.sleep(0)) BEFORE recording
        # a first-token timestamp that is actually earlier than all of
        # 0-3's. A watcher that freezes the cohort as soon as it sees K=4
        # events, without debouncing across a yield, would wrongly settle
        # on {0,1,2,3} and permanently miss request 4's legitimately
        # earlier timestamp.
        n, k, trigger_after = 5, 4, 16
        sig = _ProgressSignal()
        progress = {i: TokenProgress(request_index=i, signal=sig) for i in range(n)}
        first_token_events = {i: asyncio.Event() for i in range(n)}
        victim_tasks: dict[int, asyncio.Task] = {}

        async def immediate_task(idx: int, ns: int) -> dict:
            progress[idx].record(1, ns)
            first_token_events[idx].set()
            await asyncio.sleep(10)
            return {"status": "complete"}

        async def late_scheduled_but_earlier_task(idx: int, ns: int) -> dict:
            await asyncio.sleep(0)  # simulates "ready" but not-yet-run
            progress[idx].record(1, ns)
            first_token_events[idx].set()
            await asyncio.sleep(10)
            return {"status": "complete"}

        victim_tasks[0] = asyncio.ensure_future(immediate_task(0, 100))
        victim_tasks[1] = asyncio.ensure_future(immediate_task(1, 101))
        victim_tasks[2] = asyncio.ensure_future(immediate_task(2, 102))
        victim_tasks[3] = asyncio.ensure_future(immediate_task(3, 103))
        victim_tasks[4] = asyncio.ensure_future(late_scheduled_but_earlier_task(4, 50))

        result = await watch_dynamic_cohort_and_trigger(
            k=k, trigger_after_decode_tokens=trigger_after,
            first_token_events=first_token_events, progress_by_index=progress,
            victim_tasks=victim_tasks, request_status_ok=request_status_ok,
            cohort_freeze_timeout_s=2.0, trigger_timeout_s=0.05, signal=sig,
        )
        for t in victim_tasks.values():
            if not t.done():
                t.cancel()
        await asyncio.gather(*victim_tasks.values(), return_exceptions=True)
        return result

    res1 = asyncio.run(happy_path())
    check("phase2: happy path returns 'ok'", res1.status == DYNAMIC_TRIGGER_OK, res1.detail)
    check(
        "phase2: logical trigger equals the max of the cohort's 16th-token timestamps",
        res1.logical_trigger_ns == 5900,
        str(res1.logical_trigger_ns),
    )
    check(
        "phase2: active cohort correctly frozen as the data-driven set",
        res1.active_indices == frozenset({0, 1, 3, 4}),
    )

    res2 = asyncio.run(noncohort_leak())
    check(
        "phase2: a non-cohort token before the logical trigger invalidates the episode",
        res2.status == DYNAMIC_TRIGGER_NONCOHORT_LEAK,
        res2.detail,
    )

    res3 = asyncio.run(cohort_freeze_timeout_case())
    check(
        "phase2: insufficient first-token events within the timeout produce a clean status, no hang",
        res3.status == DYNAMIC_TRIGGER_COHORT_TIMEOUT,
        res3.status,
    )

    res4 = asyncio.run(completed_before_trigger_case())
    check(
        "issue4: cohort member whose completion_ns <= logical_trigger_ns is caught by timestamp, not task.done()",
        res4.status == DYNAMIC_TRIGGER_PRETRIGGER_FAILURE,
        f"status={res4.status} detail={res4.detail}",
    )

    res5, elapsed5 = asyncio.run(no_poll_delay_case())
    check(
        "issue1: watcher returns 'ok' once K first-token events are recorded",
        res5.status == DYNAMIC_TRIGGER_OK,
        res5.detail,
    )
    check(
        f"issue1: no 50ms-poll-granularity delay (elapsed={elapsed5*1000:.1f}ms, expected well under 50ms of "
        f"slack beyond the ~10ms the test itself sleeps)",
        elapsed5 < 0.035,
        f"elapsed={elapsed5*1000:.2f}ms",
    )

    res6 = asyncio.run(cohort_freeze_race_case())
    check(
        "issue2: cohort-freeze debounce catches a sibling that was ready-but-not-yet-scheduled "
        "and correctly includes it as the actual earliest member",
        res6.active_indices == frozenset({4, 0, 1, 2}),
        f"active_indices={res6.active_indices}",
    )
    check(
        "issue2: cohort_freeze_ns reflects the debounced (correct) cohort, not the premature one",
        res6.cohort_freeze_ns == 102,
        f"cohort_freeze_ns={res6.cohort_freeze_ns}",
    )

    return checks


def run_self_test() -> int:
    checks = _selftest_phase1() + _selftest_phase2()
    print("_active_cohort.py self-test")
    print("=" * 78)
    all_ok = True
    for name, ok, detail in checks:
        if not ok:
            all_ok = False
        line = f"[{'OK' if ok else 'FAIL'}] {name}"
        if detail and not ok:
            line += f" -- {detail}"
        print(line)
    print("=" * 78)
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(run_self_test())
