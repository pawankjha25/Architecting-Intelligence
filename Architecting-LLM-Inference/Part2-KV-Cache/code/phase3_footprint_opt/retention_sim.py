"""
Shared retention-policy simulation harness for E06/E07/E08.

IMPORTANT HONESTY NOTE: none of this loads a real model. There is no GPU,
torch, or transformers available in the environment these were authored in,
and no internet access to download model weights. Everything below is a
SYNTHETIC PROXY for the real phenomena described in the article:

  - "retrieval accuracy" = whether a synthetic marker token ("needle") placed
    at a known position is still present in the cache once generation
    reaches the end of the sequence, averaged over many needle positions.
    This is a stand-in for real long-context retrieval (e.g. needle-in-
    haystack evals on a real model) -- it tests the RETENTION POLICY, not
    an actual model's attention behavior.

  - "attention scores" for the heavy-hitter (H2O-style) policy are
    synthetically assigned, not measured from a real forward pass. We use
    them to demonstrate the MECHANISM (retain by cumulative importance,
    not just recency), not to reproduce any paper's exact numbers.

  - "collapse score" in E07 is a synthetic proxy for the real StreamingLLM
    finding that dropping the first few tokens destabilizes attention
    (their "attention sink" observation) -- it is NOT a measured attention
    entropy from a real model.

Anyone running these against a real model should replace `run_needle_test`
with an actual generation + retrieval-accuracy eval, and replace the
synthetic score assignment with real per-token attention statistics.
"""

import random
from dataclasses import dataclass


@dataclass
class RetentionResult:
    policy_name: str
    seq_len: int
    cache_budget: int
    needle_survival_rate: float   # fraction of trials where the needle was still cached at the end
    avg_cache_size: float


# -- Retention policies ---------------------------------------------------------
# Each policy is a function: (seq_len, cache_budget, needle_pos, scores) -> bool retained?
# "scores" is a synthetic importance signal per position (only used by heavy-hitter).

def sliding_window_retained(seq_len: int, cache_budget: int, position: int) -> bool:
    """Retained iff within the last `cache_budget` positions."""
    return position >= seq_len - cache_budget


def attention_sink_retained(seq_len: int, cache_budget: int, position: int, num_sinks: int = 4) -> bool:
    """Retained iff among the first `num_sinks` OR within the trailing window."""
    window = cache_budget - num_sinks
    return position < num_sinks or position >= seq_len - window


def heavy_hitter_retained(seq_len: int, cache_budget: int, position: int, scores: list, recent_protect: int = 16) -> bool:
    """
    H2O-style: always keep the most recent `recent_protect` tokens, and fill
    the remaining budget with the highest-cumulative-score tokens seen so far.
    """
    if position >= seq_len - recent_protect:
        return True
    remaining_budget = cache_budget - recent_protect
    if remaining_budget <= 0:
        return False
    # rank all non-recent positions by score, keep the top `remaining_budget`
    non_recent = [p for p in range(seq_len - recent_protect) ]
    ranked = sorted(non_recent, key=lambda p: scores[p], reverse=True)
    kept = set(ranked[:remaining_budget])
    return position in kept


def fifo_retained(seq_len: int, cache_budget: int, position: int) -> bool:
    """Pure recency, identical mechanism to sliding_window but named separately for clarity in E08's comparison table."""
    return sliding_window_retained(seq_len, cache_budget, position)


# -- Needle test ------------------------------------------------------------------

def run_needle_test(
    policy_fn,
    seq_len: int,
    cache_budget: int,
    num_trials: int = 500,
    seed: int = 42,
    needle_score_boost: float = 0.0,
    **policy_kwargs,
) -> float:
    """
    Places a needle at a random position in [0, seq_len) for each trial and
    checks whether the given policy retains it once generation reaches the
    end of the sequence. Returns the survival rate across trials.

    `needle_score_boost`: if > 0, the needle's synthetic importance score is
    boosted (simulating "this token was referenced heavily earlier in
    generation") -- used by E08 to show heavy-hitter retention succeeding
    where pure recency fails.
    """
    rng = random.Random(seed)
    survived = 0

    for _ in range(num_trials):
        needle_pos = rng.randint(0, seq_len - 1)

        # synthetic importance scores: mostly random/low, with the needle
        # possibly boosted to simulate "this token turned out to matter"
        scores = [rng.random() * 0.3 for _ in range(seq_len)]
        scores[needle_pos] += needle_score_boost

        if "scores" in policy_fn.__code__.co_varnames:
            retained = policy_fn(seq_len, cache_budget, needle_pos, scores=scores, **policy_kwargs)
        else:
            retained = policy_fn(seq_len, cache_budget, needle_pos, **policy_kwargs)

        survived += int(retained)

    return survived / num_trials
