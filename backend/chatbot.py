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
- Built by Sneha Shaw (a student/internship capstone project). If asked who made
  it / the developer / author / creator, say it was built by Sneha Shaw.

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
# Deterministic offline brain (used when there is no API key).
#
# A keyword-scoring matcher over an intent table — conversational (greetings,
# thanks, small talk) AND knowledgeable about every part of the project. Each
# intent is (keywords, answer); multi-word keywords score 2, single words score
# 1 (whole-word match so "hi" doesn't fire on "this"). Highest score wins.
# ---------------------------------------------------------------------------

_INTENTS = [
    # ---- conversational ----
    (["hi", "hello", "hey", "hii", "heya", "yo", "namaste", "namaskar",
      "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola"],
     "Hi! 👋 I'm the help assistant for this counterfeit-currency detector. "
     "Ask me how it works, how to run it, what a verdict means, or about any "
     "feature (heatmap, Hindi, the Security Pattern Studio, and more)."),

    (["how are you", "how r u", "how's it going", "hows it going", "what's up",
      "whats up", "sup", "how do you do"],
     "Doing great and ready to help! 🙂 Ask me anything about how this "
     "counterfeit-note detector works or how to use it."),

    (["thanks", "thank you", "thankyou", "thx", "ty", "appreciate", "great help"],
     "You're welcome! 😊 Happy to help — ask me anything else about the detector."),

    (["bye", "goodbye", "see you", "see ya", "cya", "good night", "gn"],
     "Bye! 👋 Stay safe — always verify a suspicious note by hand or at a bank."),

    (["ok", "okay", "cool", "nice", "great", "awesome", "got it", "understood", "fine"],
     "👍 Anything else you'd like to know about the detector?"),

    (["who are you", "what are you", "your name", "who r u", "are you a bot",
      "are you ai", "are you human", "who am i talking"],
     "I'm the built-in help assistant for this project — a counterfeit Indian-"
     "currency detector. I explain how the app works and how to use it. I'm not "
     "a general chatbot, so I stick to this project."),

    (["what can you do", "what can you help", "how can you help", "capabilities",
      "what do you do", "commands", "options", "menu", "help"],
     "I can explain: how the detector works and how to run it; what each verdict "
     "(REAL / SUSPICIOUS / FAKE / UNVERIFIED) means; every security check; the "
     "extra features (Explain-with-AI, Listen, Hindi, PDF report, live camera, "
     "AI heatmap, tamper check, Security Pattern Studio); how accurate it is; and "
     "its honest limitations. Just ask!"),

    # ---- project overview ----
    (["what is this", "what is the project", "what does this do", "what does it do",
      "about this", "about the project", "purpose", "what is the app", "explain the project",
      "tell me about the project", "tell me about this"],
     "It's a counterfeit Indian-banknote detector. You upload (or camera-capture) "
     "a photo of a note and it returns an honest verdict — REAL, SUSPICIOUS, FAKE, "
     "or UNVERIFIED — backed by ~15 forensic checks, a deep-learning classifier, "
     "and a classical-ML second opinion. It's a screening aid, not a guarantee, "
     "and it runs entirely locally."),

    (["run", "start", "launch", "install", "setup", "set up", "how to use",
      "open the app", "get started"],
     "Run it locally with: `powershell -ExecutionPolicy Bypass -File "
     "scripts\\run_local.ps1` (opens the FastAPI backend on port 8000 + the "
     "Next.js frontend on http://localhost:3000). Manually: activate the venv, "
     "run `uvicorn backend.main:app --port 8000`, then `cd frontend && npm run "
     "dev`. To use it: upload or camera-capture a note photo and press Detect "
     "Currency. See docs/SETUP_COMMANDS.txt."),

    # ---- verdicts ----
    (["real", "genuine", "authentic"],
     "REAL means most security checks passed and the note is likely genuine. "
     "It's a screening aid, not a guarantee — confirm by hand if the note matters."),

    (["fake", "counterfeit", "duplicate note"],
     "FAKE means one or more important checks failed (structure, colour, or "
     "proportions look wrong). Treat the note with caution and verify it by hand "
     "— software on a phone photo can still be wrong."),

    (["suspicious"],
     "SUSPICIOUS means mixed signals — not confidently genuine, not clearly fake. "
     "Verify it by hand: tilt for the colour-shift, hold to light for the "
     "watermark, and feel the raised print before accepting it."),

    (["unverified", "can't verify", "cant verify", "could not verify", "retake",
      "blurry", "too unclear", "low res", "low-res", "why retake"],
     "UNVERIFIED means the photo was too unclear/low-resolution to actually read "
     "the note, so the system honestly refuses to guess. Retake it: fill the "
     "frame with the note, good even light (no glare), keep it flat, and make "
     "sure the serial numbers are sharp."),

    # ---- how it works ----
    (["how does it work", "how do you detect", "how it works", "what checks",
      "forensic checks", "how do you know", "how does detection", "process",
      "pipeline", "how are you detecting"],
     "Two engines run together and are fused into one verdict: (1) a MobileNetV2 "
     "deep-learning classifier, and (2) ~15 forensic checks (serial, "
     "denomination, size & shape, watermark, security thread, colour, portrait, "
     "micro-print, serial typography, bleed lines, ID mark, UV proxy, tamper, "
     "structure). A classical-ML model gives a second opinion. Hard gates "
     "(structure / colour / size) can veto a REAL. Each check returns PASS, FAIL, "
     "or INFO ('couldn't assess')."),

    (["accurate", "accuracy", "reliable", "how good", "trust", "trustworthy",
      "is it correct", "can i trust", "is it perfect", "how reliable"],
     "It's a screening aid, not a guarantee. In evaluation it had ~0% false "
     "positives (never wrongly rejected a genuine note) and flagged most fakes, "
     "but some high-quality fakes still pass. Accuracy is limited mainly by a "
     "small (~65-image) dataset — more data, especially real physical fakes, is "
     "the biggest improvement lever."),

    # ---- coverage ----
    (["denomination", "which notes", "what notes", "supported notes", "which value",
      "10", "20", "50", "100", "200", "500", "2000"],
     "It targets the Mahatma Gandhi New Series — ₹10, ₹20, ₹50, ₹100, ₹200, ₹500 "
     "and ₹2000. It reads the value with OCR and cross-checks the note's colour "
     "palette (e.g. ₹100 is lavender, ₹500 stone, ₹2000 magenta)."),

    (["which currency", "what currency", "dollar", "euro", "other countries",
      "international", "us notes", "foreign"],
     "Indian Rupees only — the security features and dimensions are RBI-specific. "
     "Other currencies aren't supported."),

    # ---- individual checks ----
    (["serial number", "serial"],
     "The serial-number check reads both serials with EasyOCR. There's also a "
     "'serial typography' check for the RBI feature where the digits grow in size "
     "left-to-right on a genuine note."),

    (["serial typography", "typography", "number grows", "varies with size",
      "ascending", "digits grow"],
     "On genuine notes the serial digits increase in height from left to right. "
     "This check segments the digits and fits a line through their heights — a "
     "flat (non-ascending) serial is a fake signal. It's the brief's 'number "
     "varies with size' feature."),

    (["watermark"],
     "The watermark check measures the gradient/variance in the Gandhi watermark "
     "window. A genuine note shows the portrait + electrotype denomination when "
     "held to light."),

    (["security thread", "thread"],
     "The security-thread check looks for the vertical thread running through the "
     "note (it handles the modern 'windowed' dashed thread). On a real note the "
     "thread reads 'भारत' and 'RBI' and shifts colour when tilted."),

    (["proportion", "size", "shape", "aspect", "dimension", "stretched", "ratio"],
     "The size-and-shape check measures the note's aspect ratio and compares it "
     "to the official RBI mm dimensions for that denomination. A stretched image "
     "or a fake on wrong-size paper shows up as a large deviation."),

    (["colour", "color", "palette", "hologram", "tint", "colour variation"],
     "The colour check measures hue spread and saturation. Genuine notes have a "
     "rich, denomination-specific palette; washed-out or wrong colours are a fake "
     "signal. It's a hard gate that can veto a REAL verdict."),

    (["uv", "ultraviolet", "uv light"],
     "There's no UV hardware here, so the UV check is an honest visible-light "
     "*proxy* — it looks for bright reactive-ink-like patches in normal light and "
     "is shown for information only. It never decides the verdict by itself."),

    (["micro", "microprint", "micro-print", "micro lettering", "tiny print",
      "fine print", "lettering"],
     "Genuine notes carry fine micro-lettering ('RBI' + the denomination). This "
     "check measures fine-print sharpness; it flags clearly lost/blurred print "
     "but stays INFO when the photo can't resolve it — it never gives a false "
     "pass."),

    (["bleed line", "bleed lines", "edge lines", "angular lines"],
     "Bleed lines are the raised angled lines at the note's edges; the count "
     "differs by denomination (₹100→4, ₹500→5, ₹2000→7). This check counts them "
     "when the photo is sharp enough — it's informational, never a false fail."),

    (["identification mark", "id mark", "intaglio", "raised mark", "touch feature",
      "tactile"],
     "The identification mark is a raised, tactile shape (triangle/circle/etc., "
     "differs by denomination) that the visually-impaired can feel. The app "
     "checks the region and tells you which shape a genuine note should have so "
     "you can confirm by touch."),

    (["gandhi", "face", "portrait"],
     "The portrait check looks for the Mahatma Gandhi figure. Because face "
     "detection is unreliable on a stylised low-res portrait, a miss is reported "
     "as INFO (not counted against the note) rather than a false fail."),

    (["tamper", "ela", "edited", "photoshop", "photoshopped", "spliced",
      "digital editing", "manipulated", "doctored"],
     "The digital-tamper (ELA) check re-saves the image and looks at where the "
     "JPEG error concentrates — a localized hot-spot can indicate a pasted/edited "
     "region. It catches editing of the IMAGE (distinct from a physical fake) and "
     "is informational only, never driving the verdict."),

    (["heatmap", "grad-cam", "gradcam", "grad cam", "where the ai looked",
      "attention", "ai look", "what the model sees"],
     "The 'Show AI heatmap' overlay is a Grad-CAM map — it highlights where the "
     "MobileNetV2 model focused when scoring the note (red = most influential). "
     "It's an explainability view of the deep model."),

    # ---- ML ----
    (["model", "cnn", "mobilenet", "neural", "deep learning"],
     "The deep model is MobileNetV2 (a CNN) that scores the whole image REAL/FAKE. "
     "It's the weakest single component on this small dataset, which is why "
     "forensic checks are weighted more (0.6 vs 0.4) and the verdict has hard "
     "gates."),

    (["svm", "random forest", "classical", "knn", "logistic", "second opinion",
      "various machine learning", "machine learning techniques", "techniques"],
     "Besides the CNN, classical models (SVM, Random Forest, KNN, Logistic "
     "Regression) are trained on hand-crafted visual features (LBP texture, "
     "colour, structure, forensic signals) and shown as a 'second opinion'. "
     "They're benchmarked in docs/BENCHMARK.md and actually beat the CNN on this "
     "data."),

    # ---- features ----
    (["guilloche", "guilloché", "security pattern", "pattern studio",
      "generate pattern", "seed"],
     "The Security Pattern Studio procedurally generates guilloché art — the "
     "woven mathematical line patterns real security printing uses to resist "
     "copying. Give it a seed (any word/number) and it draws a unique pattern; "
     "the same seed always makes the same one. It's abstract art demonstrating "
     "anti-copy design — it does NOT generate currency."),

    (["explain with ai", "explanation", "explain", "plain language", "summary",
      "why this verdict"],
     "'Explain with AI' turns the technical result into a plain-language summary "
     "plus manual checks you can do by hand. It uses Claude when an API key is "
     "set, and a built-in template otherwise — so it always works."),

    (["listen", "read aloud", "speak", "voice", "text to speech", "tts",
      "accessibility", "blind", "visually impaired", "audio"],
     "The 'Listen' button reads the result aloud (browser text-to-speech) in the "
     "selected language — an accessibility feature for visually-impaired users, "
     "who the security features are also designed for."),

    (["hindi", "language", "translate", "हिंदी", "bilingual", "english"],
     "Use the English / हिंदी toggle near the top. It switches the verdict and "
     "the AI explanation to Hindi, and 'Listen' will read it aloud in Hindi too."),

    (["pdf", "report", "download", "print", "save report", "document"],
     "After a result, click 'Download report (PDF)'. It builds a clean one-page "
     "report (verdict, denomination, confidence, serial, the note image, the full "
     "findings table) that you can save as a PDF from the print dialog."),

    (["camera", "webcam", "capture", "live", "take a photo", "scan"],
     "Click 'Use camera' → 'Enable camera', place the note inside the green box "
     "and fill it, then Capture. Only the box is captured, so the note (not the "
     "room) is what gets analysed. Fill the box in good light for the best read."),

    (["remove", "clear photo", "clear image", "delete photo", "start over",
      "reset", "change photo"],
     "Use the red 'Remove' button on the top-right of the image preview — it "
     "clears the photo and result so you can pick or capture a new one without "
     "refreshing."),

    # ---- meta ----
    (["offline", "internet", "api key", "anthropic", "claude key",
      "without internet", "need internet", "online"],
     "Everything runs locally and works offline. The AI explanation and this "
     "chat use Claude when an ANTHROPIC_API_KEY is set, but both fall back to a "
     "built-in version with no key — so nothing requires internet."),

    (["legal", "illegal", "is it legal", "generate fake", "create real note",
      "make uncloneable", "print money"],
     "This is a detection/verification tool — it only checks notes, it can't and "
     "won't generate currency. The Security Pattern Studio makes abstract "
     "anti-copy *art*, never note imagery. Using the detector to verify notes is "
     "perfectly legitimate."),

    (["limitation", "limit", "weakness", "can it be wrong", "wrong", "mistake",
      "false positive", "false negative", "fails", "drawback"],
     "Honest limits: it's a phone-photo screening aid, so it can be wrong — "
     "especially on high-quality physical fakes (some still pass) or poorly-framed "
     "photos. It's Indian Rupees only, single-side, and UV is just a visible-light "
     "proxy. The biggest lever for improvement is more training data."),

    (["dataset", "training data", "how many images", "trained on", "data size",
      "how much data"],
     "It's trained/evaluated on ~65 labelled images (≈42 genuine / 23 fake), a "
     "mix of project fixtures and a public Kaggle set. That small size is the "
     "main accuracy ceiling — adding real physical-counterfeit images is the top "
     "improvement."),

    (["tech stack", "built with", "technology", "framework", "fastapi",
      "next.js", "react", "python", "what language", "stack"],
     "Backend: Python + FastAPI, OpenCV for the forensic checks, EasyOCR for "
     "text, TensorFlow/Keras (MobileNetV2) + scikit-learn for ML, and Claude for "
     "explanations. Frontend: Next.js + React + Tailwind."),

    (["privacy", "store my", "save my image", "keep my photo", "data privacy",
      "upload my", "is my data safe"],
     "It runs locally on your machine — your image is processed on the backend "
     "you're running and isn't stored or sent anywhere (the only optional "
     "external call is to Claude for the explanation/chat if you set an API key)."),

    (["who made", "developer", "author", "internship", "creator", "team",
      "project by", "made this", "sneha", "shaw", "who built", "whose project",
      "who created", "who developed"],
     "This project was built by Sneha Shaw — a student/internship capstone, "
     "'Counterfeit Bank Currency Detection with Various Machine Learning "
     "Techniques' for Indian Rupees."),

    (["source code", "github", "repo", "repository", "code", "open source"],
     "The code lives in this project's repository; the README and docs/REPORT.md "
     "cover the architecture, methodology and results."),

    (["said 50", "said wrong", "wrong denomination", "misread", "it said", "showed wrong",
      "it is real but"],
     "A wrong read almost always means the note wasn't well-framed — small, "
     "tilted, or with a busy background, so it couldn't be isolated. Re-shoot with "
     "the note filling the frame (or fill the green box in camera mode) in good "
     "light, and it should read correctly."),
]

_HELP_MENU = (
    "I'm the help assistant for this counterfeit-detection project. You can ask "
    "me things like:\n"
    "• How does it work? / What checks does it do?\n"
    "• What does SUSPICIOUS / UNVERIFIED mean?\n"
    "• How accurate is it? / What are its limits?\n"
    "• What is the AI heatmap / Security Pattern Studio / Listen / Hindi?\n"
    "• How do I run it?"
)

_FALLBACK = (
    "I'm not sure about that one — I only know about this counterfeit-detection "
    "project.\n\n" + _HELP_MENU
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


def _score(q, keywords):
    """Score how well a question matches an intent's keywords.
    Multi-word phrases score 2; single words score 1 via whole-word match
    (so 'hi' doesn't fire on 'this')."""
    total = 0
    for kw in keywords:
        if " " in kw:
            if kw in q:
                total += 2
        elif re.search(r"\b" + re.escape(kw) + r"\b", q):
            total += 1
    return total


def _template_answer(question):
    """Deterministic conversational matcher. Always returns a useful string."""
    q = (question or "").strip().lower()
    if not q:
        return _HELP_MENU
    if _REFUSE_PATTERN.search(q):
        return _REFUSAL

    best_answer, best_score = None, 0
    for keywords, answer in _INTENTS:
        s = _score(q, keywords)
        if s > best_score:
            best_answer, best_score = answer, s

    return best_answer if best_answer is not None else _FALLBACK


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
