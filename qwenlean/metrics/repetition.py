"""Literal repetition, exact cyclical loops, and a lexical redundancy proxy."""

import math
import re
from collections import Counter, defaultdict

STOP_WORDS = set("the a an and or to of in is it be this that with for i we then so".split())


def words(text):
    return re.findall(r"[\w]+|[-+*/=<>]", text.casefold())


def longest_repeated_span(sequence):
    """Exact longest non-overlapping repeated span, measured in supplied units."""

    def repeated(n):
        first = {}
        for i in range(len(sequence) - n + 1):
            key = tuple(sequence[i : i + n])
            if key in first and i - first[key] >= n:
                return True
            first.setdefault(key, i)
        return False

    lo, hi = 0, len(sequence) // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if repeated(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def suffix_loop(ids, min_period=24, max_period=128, repeats=3):
    """Strong exact suffix-cycle evidence; no semantic or approximate claims."""
    for period in range(min_period, min(max_period, len(ids) // repeats) + 1):
        block = ids[-period:]
        if all(ids[-period * (k + 1) : -period * k] == block for k in range(1, repeats)):
            return {
                "period_tokens": period,
                "repeats": repeats,
                "evidence_start_token": len(ids) - period * repeats,
                "end_token": len(ids),
            }
    return None


def find_loop(ids, **thresholds):
    # Check every endpoint offline, so detection is independent of runtime polling cadence.
    for endpoint in range(
        thresholds.get("min_period", 24) * thresholds.get("repeats", 3), len(ids) + 1
    ):
        evidence = suffix_loop(ids[max(0, endpoint - 512) : endpoint], **thresholds)
        if evidence:
            offset = max(0, endpoint - 512)
            evidence["evidence_start_token"] += offset
            evidence["end_token"] += offset
            return evidence
    return None


def lexical_redundancy(text, threshold=0.90, min_words=10):
    """Local bag-of-word cosine on sentences/lines. Proxy, not semantic equivalence."""
    chunks = [words(c) for c in re.split(r"\n+|(?<=[.!?])\s+", text)]
    chunks = [c for c in chunks if len(c) >= min_words]
    vectors = [Counter(w for w in c if w not in STOP_WORDS) for c in chunks]
    duplicate_tokens, pairs = 0, []
    for i, v in enumerate(vectors):
        norm = math.sqrt(sum(n * n for n in v.values()))
        for j, other in enumerate(vectors[:i]):
            denominator = norm * math.sqrt(sum(n * n for n in other.values()))
            cosine = sum(n * other[w] for w, n in v.items()) / denominator if denominator else 0
            if cosine >= threshold:
                duplicate_tokens += len(chunks[i])
                pairs.append({"earlier_chunk": j, "later_chunk": i, "cosine": cosine})
                break
    return {
        "method": "content_word_cosine_proxy_v1",
        "threshold": threshold,
        "eligible_chunks": len(chunks),
        "near_duplicate_chunks": len(pairs),
        "density": duplicate_tokens / max(1, sum(map(len, chunks))),
        "pairs": pairs,
    }


def repetition_metrics(text, ids, config=None):
    config = config or {}
    sequence = words(text)
    repeated = {}
    covered = set()
    for n in config.get("ngrams", [5, 8, 12]):
        occurrences = defaultdict(list)
        for i in range(len(sequence) - n + 1):
            span = sequence[i : i + n]
            if sum(w not in STOP_WORDS for w in span) < max(2, n // 3):
                continue
            occurrences[tuple(span)].append(i)
        repeats = {key: starts for key, starts in occurrences.items() if len(starts) > 1}
        redundant = sum(len(starts) - 1 for starts in repeats.values())
        eligible = sum(map(len, occurrences.values()))
        repeated[str(n)] = {
            "repeated_unique_ngrams": len(repeats),
            "redundant_occurrences": redundant,
            "redundant_fraction": redundant / max(1, eligible),
        }
        if n >= 8:
            for starts in repeats.values():
                for start in starts[1:]:
                    covered.update(range(start, start + n))
    return {
        "ngrams": repeated,
        "word_count": len(sequence),
        "longest_repeated_span_words": longest_repeated_span(sequence),
        "repetition_density": len(covered) / max(1, len(sequence)),
        "loop_evidence": find_loop(ids, **config.get("loop", {})),
        "lexical_redundancy": lexical_redundancy(text, **config.get("lexical", {})),
    }
