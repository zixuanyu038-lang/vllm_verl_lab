"""One shared strict scoring rule for HTTP evaluation and verl."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    match = re.fullmatch(r"\s*####\s*(-?\d+)\s*", solution_str)
    format_ok = match is not None
    correct = format_ok and int(match.group(1)) == int(ground_truth)
    return {"score": float(correct), "format_ok": float(format_ok)}
