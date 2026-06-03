"""
Phase L — project help chatbot.

A small assistant that answers "how does this work / how do I run it / what does
this verdict mean" questions about the Counterfeit Currency Detection system.

Design (same graceful pattern as backend/genai.py):
  - answer(question, history) -> {reply, source}
  - When ANTHROPIC_API_KEY is set: uses Claude (Haiku — cheap, fast, this is a
    short grounded Q&A) with a cached system prompt that contains a curated
    knowledge base about THIS project, so answers are accurate and on-topic.
  - When no key (the usual local-demo case): a deterministic keyword FAQ matcher
    answers the common questions, so the chatbot still works live in a demo
    with no internet / no key.
  - Never raises. The Anthropic SDK is imported lazily.

Scope guard: it only answers questions about this project. It is NOT a general
chatbot and must not give instructions for producing counterfeit money.
"""

import os
import re

_MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Curated project knowledge — the single source of truth the assistant uses.
# Kept in sync with PROJECT_SCOPE.md / STATUS.md. Both the live Claude path
# (as the system prompt) and the offline FAQ path draw from these facts.
# ---------------------------------------------------------------------------

_PROJECT_KNOWLEDGE = """
You are the built-in help assistant for the "Counterfeit Currency Detection
System" — a final-year/internship project that checks whether a photo of an
Indian Rupee banknote is likely genuine or fake. You answer questions about how
the project works and how to use it. Be friendly, concise, and plain-spoken.

WHAT IT IS
- A local web app (no cloud). A user uploads a phone photo of an Indian banknote
  and gets a verdict: REAL, SUSPICIOUS, FAKE, or UNVERIFIED ("can't verify").
- It is a screening aid, NOT a guarantee. It can be wrong, especially on
  high-quality fakes or poor photos.

HOW THE VERDICT IS MADE
- Two engines run on the image and are fused:
  1. A deep-learning image classifier (MobileNetV2 CNN) → REAL/FAKE + confidence.
  2. A forensic pipeline of ~11 independent OpenCV/OCR checks.
  - Plus a classical-ML "second opinion" (SVM / Random Forest / etc.) shown for
    transparency. The forensic checks are weighted more than the CNN
    (0.6 vs 0.4) because the CNN is brittle on unusual inputs.
- Hard gates: if the structure, colour palette, or proportions are clearly
  wrong, the note is failed regardless of score.

THE VERDICTS, IN PLAIN TERMS
- REAL: most checks passed; likely genuine (still confirm by hand if it matters).
- SUSPICIOUS: mixed signals; verify by hand before accepting.
- FAKE: one or more important checks failed; treat with caution.
- UNVERIFIED: the photo was too unclear/low-resolution to actually read the
  note, so the system honestly refuses to guess — retake the photo.

THE FORENSIC CHECKS (plain meaning)
- Serial number (OCR): reads the note's serial via EasyOCR.
- Denomination: reads the note's value (₹100, ₹500, …).
- Size & shape (proportions): compares the note's measured aspect ratio to the
  official RBI dimensions for that denomination (catches stretched/wrong-size).
- Watermark, Security thread, Colour palette, Portrait (Gandhi face),
  Micro-print, Serial typography ("the number grows in size left→right"),
  UV proxy (visible-light only — no UV hardware), and overall structure.
- Each returns PASS, FAIL, or INFO. INFO means "couldn't assess" — never a fake
  pass/fail. The system is deliberately honest about what it can't measure.

THE GENERATIVE-AI FEATURES
- "Explain with AI": turns the technical result into a plain-language summary +
  manual checks you can do by hand (accessibility-first; can be read aloud).
- "Security Pattern Studio": procedurally generates guilloché art — the woven
  mathematical line patterns real security printing uses to resist copying. It
  is abstract art generated from a seed (same seed → same pattern); it does NOT
  generate currency.
- This help chatbot.

HOW TO RUN IT (local)
- One command (Windows): powershell -ExecutionPolicy Bypass -File scripts\\run_local.ps1
- Or manually: activate the venv, run `uvicorn backend.main:app --port 8000`
  for the backend, and `cd frontend && npm run dev` for the UI (http://localhost:3000).
- Tests: `venv\\Scripts\\python -m pytest tests/`.

HONEST LIMITATIONS
- Accuracy is limited by the small training dataset (~65 images). On evaluation
  it had ~0% false positives (never wrongly rejected a genuine note as fake) but
  some high-quality fakes still pass. More data — especially real physical
  counterfeits — is the biggest lever for improvement.
- It only handles Indian Rupees, only from a single-side photo, and UV is a
  visible-light proxy (no special hardware).

RULES
- Only answer questions about THIS project and how to use it.
- Never explain how to produce, improve, or pass off counterfeit money. If
  asked, refuse briefly and redirect to the detection/verification purpose.
- If you don't know, say so and suggest checking the README or docs/REPORT.md.
"""

# ---------------------------------------------------------------------------
# Deterministic offline FAQ — used when there is no API key. Ordered list of
# (keyword-pattern, answer). First match wins; falls back to a help menu.
# ---------------------------------------------------------------------------

_FAQ = [
    (r"\b(run|start|launch|install|setup|set up)\b",
     "To run it locally: from the project root run "
     "`powershell -ExecutionPolicy Bypass -File scripts\\run_local.ps1` — that "
     "opens the backend (FastAPI on port 8000) and the frontend (Next.js on "
     "http://localhost:3000). Manually: activate the venv, run "
     "`uvicorn backend.main:app --port 8000`, then in another terminal "
     "`cd frontend && npm run dev`. See docs/SETUP_COMMANDS.txt for details."),

    (r"\b(unverified|can.?t verify|retake|blurry|too unclear|low.?res)\b",
     "UNVERIFIED means the photo was too unclear or low-resolution to actually "
     "read the note, so the system honestly refuses to guess rather than show a "
     "misleading result. Retake the photo: fill the frame with the note, use "
     "good even lighting (no glare), keep it flat, and make sure both serial "
     "numbers are sharp."),

    (r"\b(suspicious)\b",
     "SUSPICIOUS means the checks gave mixed signals — not confidently genuine "
     "and not clearly fake. Treat the note with caution and verify it by hand "
     "(tilt for the colour-shift, hold to light for the watermark, feel the "
     "raised print) before accepting it."),

    (r"\b(real|genuine)\b.*\b(mean|verdict)\b|\bwhat.*\breal\b",
     "REAL means most security checks passed and the note is likely genuine. "
     "It's still a screening aid, not a guarantee — confirm by hand if the note "
     "matters."),

    (r"\b(fake|counterfeit)\b.*\bmean\b|\bwhat.*\bfake\b",
     "FAKE means one or more important checks failed (e.g. structure, colour, or "
     "proportions look wrong). Treat the note with caution and verify by hand — "
     "software on a phone photo can still be wrong."),

    (r"\b(accurate|accuracy|reliable|how good|trust)\b",
     "It's a screening aid, not a guarantee. On evaluation it had ~0% false "
     "positives (it never wrongly rejected a genuine note as fake) and flagged "
     "most fakes, but some high-quality fakes still pass. Accuracy is limited by "
     "a small (~65-image) dataset — more data, especially real physical fakes, "
     "is the biggest improvement lever."),

    (r"\b(guilloch|security pattern|pattern studio)\b",
     "The Security Pattern Studio procedurally generates guilloché art — the "
     "woven mathematical line patterns real security printing uses to resist "
     "copying. You give it a seed (any word/number) and it draws a unique "
     "pattern; the same seed always makes the same pattern. It's abstract art to "
     "demonstrate anti-copy design — it does NOT generate currency."),

    (r"\b(explain with ai|explanation|plain language)\b",
     "\"Explain with AI\" turns the technical result into a plain-language "
     "summary plus manual checks you can do by hand. It's accessibility-first "
     "(the text can be read aloud). It uses Claude when an API key is set, and a "
     "built-in template otherwise."),

    (r"\b(uv|ultraviolet)\b",
     "There's no UV hardware here, so the UV check is an honest visible-light "
     "*proxy* — it looks for bright reactive-ink-like patches in normal light "
     "and is shown for information only. It never decides the verdict by itself."),

    (r"\b(proportion|size|shape|aspect|dimension)\b",
     "The size-and-shape check measures the note's aspect ratio and compares it "
     "to the official RBI dimensions for that denomination. A stretched image or "
     "a fake printed on wrong-size paper shows up as a large deviation."),

    (r"\b(serial|number)\b",
     "The serial-number check reads the note's serial with EasyOCR. There's also "
     "a 'serial typography' check for the RBI feature where the digits grow in "
     "size from left to right on a genuine note."),

    (r"\b(check|checks|feature|features|forensic|how.*work|how.*detect)\b",
     "It runs ~11 forensic checks (serial, denomination, size & shape, watermark, "
     "security thread, colour palette, portrait, micro-print, serial typography, "
     "UV proxy, structure) plus a MobileNetV2 CNN and a classical-ML second "
     "opinion, then fuses them into one verdict. Each check returns PASS, FAIL, "
     "or INFO ('couldn't assess')."),

    (r"\b(model|cnn|ml|machine learning|svm|random forest|neural)\b",
     "Two kinds of ML run together: a MobileNetV2 deep-learning image classifier, "
     "and classical models (SVM, Random Forest, KNN, Logistic Regression) trained "
     "on hand-crafted visual features. The classical models are benchmarked in "
     "docs/BENCHMARK.md and shown as a 'second opinion'."),
]

_HELP_MENU = (
    "I'm the help assistant for this counterfeit-detection project. You can ask "
    "me things like:\n"
    "• How do I run it?\n"
    "• What does SUSPICIOUS / UNVERIFIED mean?\n"
    "• What checks does it do? / How does it work?\n"
    "• How accurate is it?\n"
    "• What is the Security Pattern Studio?"
)

_REFUSAL = (
    "I can only help with using this counterfeit-detection tool and "
    "understanding its results — I can't help with making or passing fake money."
)

_REFUSE_PATTERN = re.compile(
    r"\b(make|create|print|produce|forge|improve|pass off)\b.*\b(fake|counterfeit|forged)\b"
    r"|\b(fake|counterfeit)\b.*\b(money|note|currency|cash)\b.*\b(make|create|print|produce)\b",
    re.IGNORECASE,
)


def _template_answer(question):
    """Deterministic keyword FAQ. Always returns a useful string."""
    q = (question or "").strip().lower()
    if not q:
        return _HELP_MENU
    if _REFUSE_PATTERN.search(q):
        return _REFUSAL
    for pattern, reply in _FAQ:
        if re.search(pattern, q):
            return reply
    return (
        "I'm not sure about that specific question. " + _HELP_MENU
        + "\n\nFor anything deeper, see the README or docs/REPORT.md."
    )


def llm_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def answer(question, history=None):
    """Answer a help question. Returns {reply, source}. Never raises.

    history: optional list of {role: 'user'|'assistant', content: str} for
    multi-turn context. source is 'llm' when Claude produced it, else 'template'."""

    if not isinstance(question, str) or not question.strip():
        return {"reply": _HELP_MENU, "source": "template"}

    # Safety refusal happens regardless of backend.
    if _REFUSE_PATTERN.search(question):
        return {"reply": _REFUSAL, "source": "template"}

    if not llm_available():
        return {"reply": _template_answer(question), "source": "template"}

    try:
        import anthropic

        messages = []
        if isinstance(history, list):
            for turn in history[-6:]:  # cap context to the last few turns
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()})
        messages.append({"role": "user", "content": question.strip()})

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=500,
            system=[{
                "type": "text",
                "text": _PROJECT_KNOWLEDGE,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        if not text:
            return {"reply": _template_answer(question), "source": "template"}
        return {"reply": text, "source": "llm"}
    except Exception:
        return {"reply": _template_answer(question), "source": "template"}
