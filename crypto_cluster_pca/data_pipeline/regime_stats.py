#!/usr/bin/env python3
"""
Causal (expanding-window) regime statistics.

This module replaces the primary leak found in Overfitting_Diagnosis.md item 2(ii).

The original (backtest_data_manager.py:432-477) computed `persistence`, `avg_duration`
and `frequency_percentage` over the ENTIRE sample and then broadcast each regime's
single value onto every row of that regime:

    for _, stats in regime_stats_df.iterrows():
        mask = regime_df['regime_id'] == regime_id
        regime_df.loc[mask, 'persistence'] = stats['persistence']      # full-sample
        regime_df.loc[mask, 'avg_duration'] = stats['avg_duration']    # full-sample
        regime_df.loc[mask, 'frequency_percentage'] = stats['percentage']

Those three values then drove `should_trade`, the coin-score base, the position-sizing
persistence bonus, the stop-loss persistence adjuster, and the take-profit duration
bonus. On day 1 the strategy already knew how persistent and how frequent each regime
would turn out to be across the whole year.

Here every statistic at time t uses only observations in [0, t].

Completed spells only
---------------------
The original appended the trailing, still-running spell to `durations`
(backtest_data_manager.py:455-456) — which is how regime 4 reported avg_duration = 42
from a single possibly-unfinished episode. A spell's length is unknowable until it
ends, so `avg_duration` here averages only spells that closed strictly before t.

Early-sample policy
-------------------
With fewer than MIN_COMPLETED_SPELLS closed episodes, these statistics are noise rather
than estimates. They are emitted as NaN, and the caller must decide what to do. Emitting
a confident number from one observation is precisely the failure this rebuild corrects.
"""

import numpy as np
import pandas as pd

MIN_COMPLETED_SPELLS = 2


def _spell_ids(labels: np.ndarray) -> np.ndarray:
    """Contiguous-run id for each observation."""
    return np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])


def causal_regime_stats(
    labels: pd.Series, min_spells: int = MIN_COMPLETED_SPELLS
) -> pd.DataFrame:
    """
    Expanding-window per-observation regime statistics.

    Parameters
    ----------
    labels : regime id per timestamp, in chronological order.

    Returns
    -------
    DataFrame indexed like `labels` with, for the regime active at each t and using
    only data in [0, t]:
      persistence           P(stay in regime | in regime), from transitions seen so far
      avg_duration          mean length of spells of this regime COMPLETED before t
      frequency_percentage  share of observations so far spent in this regime
      completed_spells      how many spells of this regime have closed (diagnostic)
    """
    lab = labels.to_numpy()
    n = len(lab)
    spells = _spell_ids(lab)

    # Per-regime running accumulators, updated strictly in time order.
    counts: dict = {}                      # observations seen in each regime
    stay: dict = {}                        # t->t+1 transitions that stayed
    opportunities: dict = {}               # transitions observed from each regime
    finished: dict = {}                    # lengths of CLOSED spells per regime

    persistence = np.full(n, np.nan)
    avg_duration = np.full(n, np.nan)
    freq_pct = np.full(n, np.nan)
    n_spells = np.zeros(n, dtype=int)

    cur_len = 0
    for t in range(n):
        r = lab[t]

        # --- update state with observation t (never anything after t) ---
        counts[r] = counts.get(r, 0) + 1
        if t > 0:
            prev = lab[t - 1]
            opportunities[prev] = opportunities.get(prev, 0) + 1
            if prev == r:
                stay[prev] = stay.get(prev, 0) + 1
            else:
                # the previous spell just closed — only now is its length known
                finished.setdefault(prev, []).append(cur_len)
                cur_len = 0
        cur_len += 1

        # --- emit statistics for regime r using [0, t] only ---
        opp = opportunities.get(r, 0)
        if opp > 0:
            persistence[t] = stay.get(r, 0) / opp

        done = finished.get(r, [])
        n_spells[t] = len(done)
        if len(done) >= min_spells:
            avg_duration[t] = float(np.mean(done))

        freq_pct[t] = counts[r] / (t + 1) * 100.0

    return pd.DataFrame(
        {
            "persistence": persistence,
            "avg_duration": avg_duration,
            "frequency_percentage": freq_pct,
            "completed_spells": n_spells,
        },
        index=labels.index,
    )


def full_sample_regime_stats(labels: pd.Series) -> pd.DataFrame:
    """
    The ORIGINAL leaking computation, kept for the Phase 6 write-up.

    Reproduces backtest_data_manager.py:432-477 exactly, including the trailing
    in-progress spell. Used only to quantify how large the leak was; never used to
    generate a trading decision.
    """
    lab = labels.to_numpy()
    out = {}
    for r in np.unique(lab):
        mask = lab == r
        total = mask.sum()
        transitions = ((labels.shift(1) == r) & (labels == r)).sum()
        runs, cur = [], 0
        for v in mask:
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:                      # the original counted the unfinished spell
            runs.append(cur)
        out[r] = {
            "persistence": transitions / max(total - 1, 1),
            "avg_duration": float(np.mean(runs)) if runs else 1.0,
            "frequency_percentage": total / len(lab) * 100.0,
        }
    return pd.DataFrame(
        [out[r] for r in lab], index=labels.index
    )
