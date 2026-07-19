"""
Promote reviewed 👎 feedback into the offline eval set — closing the loop
that C6 opened.

The webapp writes every downvoted answer a reviewer marks into
`data/feedback_eval_queue.json` (via the dashboard's "Vào eval" button). Once
a reviewer fills in the correct `ground_truth` for an entry, this script
appends it to `data/webapp_eval_questions.json` so the next `run_webapp_eval.py`
scores the model on the very questions users were unhappy with — turning
production complaints into regression tests (TASKLIST D4).

Entries without a `ground_truth` are skipped (nothing to score against), and
each promoted entry is marked so re-running is idempotent.

Usage:
  python scripts/promote_feedback_to_eval.py            # promote reviewed entries
  python scripts/promote_feedback_to_eval.py --list     # show queue status only
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = ROOT / "data" / "feedback_eval_queue.json"
QUESTIONS_PATH = ROOT / "data" / "webapp_eval_questions.json"

_JA_RE = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ一-鿿]")
_VI_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữựỳýỷỹỵ]", re.IGNORECASE)


def _detect_lang(text: str) -> str:
    """Cheap language tag consistent with the rest of the pipeline."""
    if _JA_RE.search(text):
        return "ja"
    if _VI_RE.search(text):
        return "vi"
    return "en"


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--list", action="store_true",
                    help="only report queue status, don't promote")
    args = ap.parse_args()

    queue = _load(QUEUE_PATH, [])
    if not queue:
        print(f"Queue empty ({QUEUE_PATH}). Nothing to do.")
        return

    reviewed = [e for e in queue if (e.get("ground_truth") or "").strip()]
    promoted_already = [e for e in queue if e.get("promoted")]
    pending_review = [e for e in queue
                      if not (e.get("ground_truth") or "").strip() and not e.get("promoted")]

    print(f"Queue: {len(queue)} total · {len(reviewed)} reviewed · "
          f"{len(promoted_already)} already promoted · {len(pending_review)} awaiting ground_truth")
    if args.list:
        for e in pending_review:
            print(f"  [awaiting] {e.get('question', '')[:70]}")
        return

    questions = _load(QUESTIONS_PATH, [])
    existing = {" ".join(str(q.get("question", "")).lower().split()) for q in questions}
    n_existing_fb = sum(1 for q in questions if str(q.get("id", "")).startswith("fb_"))

    added = 0
    for e in queue:
        gt = (e.get("ground_truth") or "").strip()
        if not gt or e.get("promoted"):
            continue
        q_text = (e.get("question") or "").strip()
        q_norm = " ".join(q_text.lower().split())
        if not q_text or q_norm in existing:
            e["promoted"] = True  # nothing to add, but don't revisit
            continue
        questions.append({
            "id": f"fb_{n_existing_fb + added:03d}",
            "lang": _detect_lang(q_text),
            "question": q_text,
            "ground_truth": gt,
            "source_title": (e.get("sources") or [""])[0],
            "origin": "feedback",  # provenance: promoted from a 👎 answer
        })
        existing.add(q_norm)
        e["promoted"] = True
        added += 1
        print(f"  + [{_detect_lang(q_text)}] {q_text[:70]}")

    if added:
        QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        QUEUE_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"\nPromoted {added} question(s) → {QUESTIONS_PATH.name} "
              f"(now {len(questions)} total). Run `python run_webapp_eval.py` to score them.")
    else:
        print("\nNothing new to promote (all reviewed entries already in the eval set).")


if __name__ == "__main__":
    main()
