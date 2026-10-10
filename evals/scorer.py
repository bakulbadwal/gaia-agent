"""Answer scorers: the official GAIA quasi-exact-match scorer, and the course API's strict match.

Two scorers, because the two places this agent is graded disagree:

1. ``question_scorer`` / ``gaia_match`` -- a faithful port of the official GAIA leaderboard scorer.

   Source (code only; no dataset or answer file was opened):
       https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
       file blob   ede7ece61d26cc93bf440a86a527163d617e7c12  (3212 bytes,
                   sha256 0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2)
       last commit 1349a17979f0aca0ee9c46cd7ec26eb2fb41102e "fix none parsing in answers",
                   2025-03-07T13:45:43Z
       Space HEAD  9f133d71362e77b3539f1514f31b9c101a545fec (2026-05-03), fetched 2026-10-10.

   Behaviour is unchanged. The only edits: ``print`` calls became ``logger.debug`` (the
   original prints a line per comparison), type hints were tidied, and the mutable default
   argument of ``split_string`` became a tuple. Known quirks are kept on purpose, because a
   port that "fixes" the reference stops being a port:
     * number ground truth: the answer only has ``$``, ``%`` and ``,`` stripped, so
       ``"17 km"`` vs ``"17"`` is wrong, while ``"$1,234.50"`` vs ``"1234.5"`` is right;
     * list ground truth (contains ``,`` or ``;``): split on both, lengths must match, order
       matters, number elements compare as floats, string elements keep punctuation;
     * string ground truth: whitespace and ASCII punctuation removed, case-folded, so
       ``"Sea gull."`` == ``"seagull"``;
     * an answer that cannot be parsed as a number becomes ``inf`` (never equal).

2. ``exact_match`` -- the course's Unit 4 scoring API (agents-course-unit4-scoring.hf.space),
   which is what decides the certificate. From its source:
       https://huggingface.co/spaces/agents-course/Unit4_scoring/blob/main/main.py
       Space HEAD 20e689ccac2c7336c39c5485a935f8860ae81aaa (2025-04-24), fetched 2026-10-10:
           if submitted.strip().lower() == ground_truth.strip().lower(): correct_count += 1
   So it is case-insensitive and trims the ends, and nothing else: ``"1,000"`` != ``"1000"``,
   ``"Paris."`` != ``"Paris"``, ``"a,b"`` != ``"a, b"``.

Reporting both shows how much accuracy hinges on formatting alone (the gap between them).
"""

from __future__ import annotations

import logging
import re
import string
import warnings
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_number_str",
    "split_string",
    "normalize_str",
    "question_scorer",
    "gaia_match",
    "exact_match",
    "score_answer",
]


# --------------------------------------------------------------------------- official GAIA port


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        logger.debug("String %s cannot be normalized to number str.", number_str)
        return float("inf")


def split_string(
    s: str,
    char_list: tuple[str, ...] | list[str] = (",", ";"),
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def question_scorer(
    model_answer: str | None,
    ground_truth: str,
) -> bool:
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if model_answer is None:
        model_answer = "None"

    # if gt is a number
    if is_float(ground_truth):
        logger.debug("Evaluating %s as a number.", model_answer)
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        logger.debug("Evaluating %s as a comma separated list.", model_answer)
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False) == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        logger.debug("Evaluating %s as a string.", model_answer)
        return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


# --------------------------------------------------------------------------- wrappers


def _as_text(x: Any) -> str | None:
    """Agents sometimes return ints/floats/lists; the leaderboard receives strings."""
    if x is None:
        return None
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    if isinstance(x, (list, tuple)):
        return ", ".join(str(v) for v in x)
    return str(x)


def gaia_match(model_answer: Any, ground_truth: Any) -> bool:
    """GAIA quasi-exact match, tolerant of non-string inputs and silent about list warnings."""
    gt = _as_text(ground_truth)
    if gt is None:
        raise ValueError("ground_truth must not be None")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return question_scorer(_as_text(model_answer), gt)


def exact_match(model_answer: Any, ground_truth: Any) -> bool:
    """The course API rule: ``str(a).strip().lower() == str(b).strip().lower()``."""
    if model_answer is None or ground_truth is None:
        return False
    return str(model_answer).strip().lower() == str(ground_truth).strip().lower()


def score_answer(model_answer: Any, ground_truth: Any) -> dict[str, bool]:
    """Both verdicts: ``{"gaia": quasi-exact, "strict": course API exact}``."""
    return {"gaia": gaia_match(model_answer, ground_truth), "strict": exact_match(model_answer, ground_truth)}
