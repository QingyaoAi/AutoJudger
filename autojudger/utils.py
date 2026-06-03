"""Common helpers: response parsing and prompt-template filling.

`parse_response` is ported from PRE (chuzhumin98/PRE) and kept behaviorally
compatible: it turns a raw LLM string into an int / float / nominal-tick value.
"""

import math
import re


def parse_response(response, parse_type, nominal_list=None, nominal_ticks=None):
    """Parse a raw LLM response into a structured value.

    parse_type:
        'int'   -> first signed integer found, else None
        'float' -> first signed decimal found, else None
        'str'   -> first nominal label found (by earliest position),
                   mapped to its tick; requires nominal_list + nominal_ticks
    """
    assert parse_type in ("int", "float", "str")
    if response is None:
        return None

    if parse_type == "int":
        nums = re.findall(r"-?\d+", response)
        return int(nums[0]) if nums else None

    if parse_type == "float":
        nums = re.findall(r"-?\d+\.?\d*", response)
        return float(nums[0]) if nums else None

    # parse_type == 'str'
    assert nominal_list is not None and nominal_ticks is not None
    appear_pos, cur_idx = math.inf, -1
    low = response.lower()
    for idx, label in enumerate(nominal_list):
        pos = low.find(label.lower())
        if pos != -1 and pos < appear_pos:
            appear_pos, cur_idx = pos, idx
    return nominal_ticks[cur_idx] if cur_idx != -1 else None


def fill_template(template, item):
    """Replace every ``{{key}}`` in ``template`` with ``str(item[key])``.

    Keys present in the template but missing from ``item`` are left untouched,
    so partially-filled templates (e.g. before responses are injected) work.
    """
    prompt = template
    for key, value in item.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt


# Default nominal mapping for pairwise A/B/tie judgments.
# Covers the common output vocabularies across PRE / Auto-PRE / CalibraEval.
PAIRWISE_NOMINAL_LIST = ["one", "two", "tie", "a", "b"]
PAIRWISE_NOMINAL_TICKS = [-1, 1, 0, -1, 1]
