"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

type ProportionValue = {
  actual_aspect: number;
  expected_aspect: number;
  deviation_pct: number;
  measurement: "quad" | "frame";
};

type ForensicCheck = {
  status: "PASS" | "FAIL" | "INFO";
  details: string;
  value?: string | number | ProportionValue | null;
};

type ForensicAnalysis = {
  structural_sanity: ForensicCheck;
  uv_light_detection: ForensicCheck;
  watermark_detection: ForensicCheck;
  ocr_serial_number: ForensicCheck;
  gandhi_face_analysis: ForensicCheck;
  security_thread_detection: ForensicCheck;
  serial_typography_analysis: ForensicCheck;
  microprint_detection: ForensicCheck;
  hologram_detection: ForensicCheck;
  denomination_classification: ForensicCheck;
  proportion_analysis: ForensicCheck;
  bleed_line_detection: ForensicCheck;
  identification_mark: ForensicCheck;
  tamper_detection: ForensicCheck;
  modular_ai_pipeline: ForensicCheck;
};

function isProportionValue(v: unknown): v is ProportionValue {
  return (
    typeof v === "object"
    && v !== null
    && "actual_aspect" in v
    && "expected_aspect" in v
    && "deviation_pct" in v
  );
}

type ClassicalModel = {
  available?: boolean;
  name?: string | null;
  verdict?: "REAL" | "FAKE" | null;
  confidence?: string | null;
  prob_genuine?: number | null;
};

type MlModels = {
  cnn: {
    verdict: "REAL" | "FAKE";
    confidence: string;
    prob_genuine: number;
  };
  classical: ClassicalModel;
  agreement: boolean | null;
};

type Verification = {
  level: "full" | "partial" | "none";
  note_located: boolean;
  resolution_px: number;
  serial_read: boolean;
  proportions_measured: boolean;
  denomination_read: boolean;
  unread: string[];
  guidance: string;
};

type Region = {
  label: string;
  polygon: [number, number][];
};

type Verdict = "REAL" | "FAKE" | "SUSPICIOUS" | "UNVERIFIED";

type PredictResponse = {
  status: "success" | "error";
  prediction?: Verdict;
  security_verdict?: "REAL" | "FAKE" | "SUSPICIOUS";
  verification_level?: "full" | "partial" | "none";
  verification?: Verification;
  guidance?: string;
  regions?: Region[];
  heatmap?: string | null;
  confidence?: string;
  raw_prediction?: number;
  model_verdict?: "REAL" | "FAKE";
  model_confidence?: string;
  forensic_score?: number;
  forensic_pass_count?: number;
  forensic_total_checks?: number;
  ml_models?: MlModels;
  forensic_analysis?: ForensicAnalysis;
  message?: string;
};

type Explanation = {
  summary: string;
  reasons: string[];
  manual_checks: string[];
  source: "llm" | "template";
};

type ExplainResponse = {
  status: "success" | "error";
  llm_available?: boolean;
  explanation?: Explanation;
  message?: string;
};

const VERDICT_COLOR: Record<string, string> = {
  REAL: "text-green-400",
  FAKE: "text-red-400",
  SUSPICIOUS: "text-yellow-400",
  UNVERIFIED: "text-orange-400",
};

// Human-facing verdict presentation. The internal verdict strings are
// engineer-speak; a normal user needs a plain headline + a one-line meaning.
const VERDICT_DISPLAY: Record<
  string,
  { headline: string; sub: string; accent: string; bg: string; border: string }
> = {
  REAL: {
    headline: "Likely Genuine",
    sub: "Most security checks passed. Not a guarantee — confirm by hand if it matters.",
    accent: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/40",
  },
  FAKE: {
    headline: "Likely Fake",
    sub: "One or more important checks failed. Treat with caution and verify by hand.",
    accent: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/40",
  },
  SUSPICIOUS: {
    headline: "Couldn't Confirm",
    sub: "Some checks were unclear. Treat as suspicious and verify by hand before accepting.",
    accent: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/40",
  },
  UNVERIFIED: {
    headline: "Can't Verify This Photo",
    sub: "The image was too unclear to read the note. Please retake the photo and try again.",
    accent: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/40",
  },
};

type Lang = "en" | "hi";

// Hindi headline + meaning per verdict (colours reused from VERDICT_DISPLAY).
const VERDICT_HEADLINE_HI: Record<string, { headline: string; sub: string }> = {
  REAL: {
    headline: "संभवतः असली",
    sub: "अधिकांश सुरक्षा जाँच पास हुईं। यह गारंटी नहीं है — ज़रूरी हो तो हाथ से जाँचें।",
  },
  FAKE: {
    headline: "संभवतः नकली",
    sub: "एक या अधिक महत्वपूर्ण जाँच विफल रहीं। सावधानी बरतें और हाथ से जाँचें।",
  },
  SUSPICIOUS: {
    headline: "पुष्टि नहीं हो सकी",
    sub: "कुछ जाँच अस्पष्ट रहीं। संदिग्ध मानें और स्वीकार करने से पहले हाथ से जाँचें।",
  },
  UNVERIFIED: {
    headline: "इस फ़ोटो की पुष्टि नहीं",
    sub: "नोट को पढ़ने के लिए फ़ोटो बहुत अस्पष्ट थी। कृपया दोबारा फ़ोटो लें।",
  },
};

// Small UI-string dictionary for the bilingual labels.
const UI: Record<string, { en: string; hi: string }> = {
  result: { en: "Result", hi: "परिणाम" },
  confidence: { en: "Confidence", hi: "विश्वास" },
  listen: { en: "Listen", hi: "सुनें" },
  stop: { en: "Stop", hi: "रोकें" },
  explainTitle: { en: "Explain with AI", hi: "AI से समझें" },
  explainCta: { en: "Explain this result", hi: "यह परिणाम समझाएँ" },
  regenerate: { en: "Regenerate", hi: "फिर से बनाएँ" },
  generating: { en: "Generating…", hi: "बना रहे हैं…" },
  summary: { en: "Summary", hi: "सारांश" },
  why: { en: "Why", hi: "कारण" },
  byHand: { en: "Check it by hand", hi: "हाथ से जाँचें" },
  retakeHow: { en: "How to get a good photo:", hi: "अच्छी फ़ोटो कैसे लें:" },
};

const t = (key: string, lang: Lang) => UI[key]?.[lang] ?? UI[key]?.en ?? key;

// Browser text-to-speech (accessibility). No-op if unsupported.
function speak(text: string, lang: Lang) {
  if (typeof window === "undefined" || !window.speechSynthesis || !text) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang === "hi" ? "hi-IN" : "en-US";
  u.rate = 0.98;
  window.speechSynthesis.speak(u);
}
function stopSpeaking() {
  if (typeof window !== "undefined" && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

// Build a self-contained, print-ready HTML report (Phase N). Opened in a new
// window that auto-triggers the browser print dialog → "Save as PDF".
function buildReportHtml(result: PredictResponse, imageDataUrl: string) {
  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const verdict = result.prediction ?? "SUSPICIOUS";
  const d = VERDICT_DISPLAY[verdict] ?? VERDICT_DISPLAY.SUSPICIOUS;
  const verdictColor = (
    { REAL: "#16a34a", FAKE: "#dc2626", SUSPICIOUS: "#ca8a04", UNVERIFIED: "#ea580c" }
  )[verdict] ?? "#444";
  const denomRaw = result.forensic_analysis?.denomination_classification?.value;
  const denom = typeof denomRaw === "string" ? `₹${denomRaw}` : "";
  const serialRaw = result.forensic_analysis?.ocr_serial_number?.value;
  const serial = typeof serialRaw === "string" ? serialRaw : "—";
  const when = new Date().toLocaleString();

  const fa = result.forensic_analysis;
  const rows = fa
    ? PLAIN_CHECKS.map((c) => {
        const st = (fa[c.key]?.status ?? "INFO") as "PASS" | "FAIL" | "INFO";
        const sym = st === "PASS" ? "✓" : st === "FAIL" ? "✗" : "—";
        const col = st === "PASS" ? "#16a34a" : st === "FAIL" ? "#dc2626" : "#999";
        const text = st === "PASS" ? c.good : st === "FAIL" ? c.fail : c.info;
        return `<tr><td style="color:${col};font-weight:700;width:1.4rem">${sym}</td>
          <td style="white-space:nowrap;padding-right:14px"><b>${esc(c.label)}</b></td>
          <td style="color:#444">${esc(text)}</td></tr>`;
      }).join("")
    : "";

  return `<!doctype html><html><head><meta charset="utf-8">
  <title>AuthentiNote report ${esc(denom)}</title>
  <style>
    *{box-sizing:border-box} body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#111;margin:0;padding:40px;max-width:820px}
    h1{font-size:20px;margin:0} .sub{color:#666;font-size:12px;margin-top:2px}
    .rule{height:3px;background:${verdictColor};border-radius:2px;margin:16px 0}
    .verdict{font-size:30px;font-weight:800;color:${verdictColor};margin:4px 0}
    .meaning{color:#444;font-size:13px;margin-bottom:14px}
    .grid{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
    .meta td{padding:3px 0;font-size:13px} .meta td:first-child{color:#777;padding-right:16px}
    img{max-width:320px;max-height:240px;border:1px solid #ddd;border-radius:8px}
    table.checks{border-collapse:collapse;margin-top:8px;width:100%} table.checks td{padding:4px 0;font-size:13px;vertical-align:top}
    h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#888;margin:22px 0 4px}
    .note{margin-top:24px;font-size:11px;color:#888;border-top:1px solid #eee;padding-top:10px}
    @media print{body{padding:0}}
  </style></head>
  <body onload="setTimeout(function(){window.print()},300)">
    <h1>AuthentiNote — Counterfeit Currency Screening Report</h1>
    <div class="sub">Generated ${esc(when)} · screening aid, not a legal determination</div>
    <div class="rule"></div>
    <div class="grid">
      <div style="flex:1;min-width:300px">
        <div class="verdict">${esc(d.headline)} ${esc(denom)}</div>
        <div class="meaning">${esc(d.sub)}</div>
        <table class="meta">
          <tr><td>Confidence</td><td><b>${esc(result.confidence ?? "—")}</b></td></tr>
          <tr><td>Serial number</td><td><b>${esc(serial)}</b></td></tr>
          <tr><td>Verification</td><td><b>${esc(result.verification_level ?? "—")}</b></td></tr>
          <tr><td>Forensic checks passed</td><td><b>${result.forensic_pass_count ?? 0} / ${result.forensic_total_checks ?? 0}</b></td></tr>
        </table>
      </div>
      ${imageDataUrl ? `<div><img src="${imageDataUrl}" alt="note"/></div>` : ""}
    </div>
    ${result.guidance ? `<div class="meaning" style="margin-top:12px">⚠ ${esc(result.guidance)}</div>` : ""}
    <h2>What we checked</h2>
    <table class="checks"><tbody>${rows}</tbody></table>
    <div class="note">This automated screening uses a phone-photo image and can be wrong, especially on
    high-quality counterfeits. Always verify suspicious notes by hand (tilt for colour-shift, hold to
    light for the watermark, feel the raised print) or at a bank. Indian Rupees only.</div>
  </body></html>`;
}

export default function Home() {

  // =====================================================
  // STATES
  // =====================================================

  const [selectedImage, setSelectedImage] =
    useState<File | null>(null);

  const [preview, setPreview] =
    useState<string>("");

  const [loading, setLoading] =
    useState(false);

  const [result, setResult] =
    useState<PredictResponse | null>(null);

  const [dragOver, setDragOver] = useState(false);

  const [lang, setLang] = useState<Lang>("en");

  const [cameraOpen, setCameraOpen] = useState(false);

  const [showHeatmap, setShowHeatmap] = useState(false);

  // =====================================================
  // HANDLE IMAGE CHANGE
  // =====================================================

  const acceptFile = (file?: File | null) => {
    if (!file || !file.type.startsWith("image/")) return;
    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setShowHeatmap(false);
  };

  const handleImageChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    acceptFile(e.target.files?.[0]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    acceptFile(e.dataTransfer.files?.[0]);
  };

  // Phase N — open a print-ready report in a new window (Save as PDF).
  const downloadReport = () => {
    if (!result) return;
    const open = (imgUrl: string) => {
      const w = window.open("", "_blank");
      if (!w) {
        alert("Allow pop-ups to download the report.");
        return;
      }
      w.document.write(buildReportHtml(result, imgUrl));
      w.document.close();
    };
    if (selectedImage) {
      const reader = new FileReader();
      reader.onloadend = () =>
        open(typeof reader.result === "string" ? reader.result : "");
      reader.onerror = () => open("");
      reader.readAsDataURL(selectedImage);
    } else {
      open("");
    }
  };

  // =====================================================
  // HANDLE PREDICTION
  // =====================================================

  const handlePrediction = async () => {

    if (!selectedImage) {

      alert("Please select an image");

      return;
    }

    try {

      setLoading(true);

      const formData = new FormData();

      formData.append(
        "file",
        selectedImage
      );

      const response = await axios.post(

        "http://127.0.0.1:8000/predict",

        formData,

        {
          headers: {
            "Content-Type":
              "multipart/form-data",
          },
        }
      );

      console.log(
        "API RESPONSE:",
        response.data
      );

      if (
        response.data.status ===
        "success"
      ) {

        setResult(response.data);

      } else {

        alert(
          response.data.message
        );
      }

    } catch (error) {

      console.log(error);

      alert(
        "Backend connection failed"
      );

    } finally {

      setLoading(false);
    }
  };

  // =====================================================
  // UI
  // =====================================================

  return (

    <main className="min-h-screen text-white flex flex-col items-center px-5 sm:px-6 py-14">

      {/* ================================================= */}
      {/* HERO */}
      {/* ================================================= */}

      <div className="w-full max-w-4xl text-center mb-10">

        <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3.5 py-1.5 text-xs font-medium text-emerald-300/90 mb-6 backdrop-blur">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Indian Rupee · Forensic + AI
        </span>

        <h1 className="text-4xl sm:text-6xl font-bold tracking-tight leading-[1.05]">
          <span className="bg-gradient-to-br from-white via-white to-emerald-200/80 bg-clip-text text-transparent">
            Counterfeit Currency
          </span>
          <br />
          <span className="bg-gradient-to-br from-emerald-300 via-emerald-400 to-teal-400 bg-clip-text text-transparent">
            Detection
          </span>
        </h1>

        <p className="text-gray-400 mt-5 max-w-2xl mx-auto leading-relaxed">
          Upload a photo of an Indian banknote. The system fuses a deep-learning
          classifier with a forensic check pipeline, shows what it found in plain
          language, and is honest when it can&apos;t be sure. A screening aid —
          not a guarantee.
        </p>

        {/* Language toggle (Phase M — accessibility/reach) */}
        <div className="mt-6 inline-flex items-center rounded-full border border-white/10 bg-white/5 p-1 text-sm">
          {(["en", "hi"] as Lang[]).map((l) => (
            <button
              key={l}
              onClick={() => { setLang(l); stopSpeaking(); }}
              className={`px-4 py-1.5 rounded-full font-medium transition-all ${
                lang === l
                  ? "bg-emerald-500 text-white shadow"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {l === "en" ? "English" : "हिंदी"}
            </button>
          ))}
        </div>

      </div>

      {/* ================================================= */}
      {/* MAIN CARD */}
      {/* ================================================= */}

      <div className="card card-hover w-full max-w-4xl p-6 sm:p-8 animate-in">

        {/* DROPZONE */}

        <input
          id="note-file"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="hidden"
        />
        <label
          htmlFor="note-file"
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`
          group flex flex-col items-center justify-center gap-3 cursor-pointer
          rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-all
          ${dragOver
            ? "border-emerald-400/70 bg-emerald-500/10"
            : "border-white/15 bg-white/[0.02] hover:border-emerald-400/40 hover:bg-white/[0.04]"}
          `}
        >
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-400/30 transition-transform group-hover:scale-105">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </span>
          <div>
            <p className="font-semibold text-gray-100">
              {selectedImage ? selectedImage.name : "Drop a banknote photo here"}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              or <span className="text-emerald-300 font-medium">browse</span>
              {" · JPG / PNG · fill the frame for best results"}
            </p>
          </div>
        </label>

        {/* Use-camera option (Phase O) */}
        <div className="mt-3 flex justify-center">
          <button
            onClick={() => setCameraOpen(true)}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
              <circle cx="12" cy="13" r="4" />
            </svg>
            Use camera
          </button>
        </div>

        {/* IMAGE PREVIEW (with detected-note overlay — Phase G.3) */}

        {preview && (

          <div className="mt-6 relative w-full overflow-hidden rounded-2xl ring-1 ring-white/10">

            <img
              src={preview}
              alt="Preview"
              className="
              w-full
              h-auto
              block
              bg-black
              "
            />

            {/* Detected-region overlay: polygons are normalised [0,1] in the
                original image's coordinates, so with the image rendered at
                full width / natural height the SVG aligns exactly. */}
            {result?.regions && result.regions.length > 0 && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
              >
                {result.regions.map((r, i) => (
                  <polygon
                    key={i}
                    points={r.polygon.map((p) => `${p[0]},${p[1]}`).join(" ")}
                    fill="rgba(34,197,94,0.12)"
                    stroke="rgb(74,222,128)"
                    strokeWidth="0.006"
                    strokeLinejoin="round"
                  />
                ))}
              </svg>
            )}

            {result?.regions && result.regions.length > 0 && (
              <span className="absolute top-3 left-3 inline-flex items-center gap-1.5 text-xs font-semibold bg-emerald-400 text-black px-2.5 py-1 rounded-lg shadow-lg">
                <span className="h-1.5 w-1.5 rounded-full bg-black/70" />
                Detected note
              </span>
            )}

            {/* Grad-CAM heatmap overlay (Phase P.2) — where the AI looked */}
            {showHeatmap && result?.heatmap && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={result.heatmap}
                alt="AI attention heatmap"
                className="absolute inset-0 w-full h-full pointer-events-none mix-blend-screen"
              />
            )}

            {result?.heatmap && (
              <button
                onClick={() => setShowHeatmap((s) => !s)}
                className={`absolute top-3 right-3 inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-lg shadow-lg transition-colors ${
                  showHeatmap
                    ? "bg-fuchsia-500 text-white"
                    : "bg-black/70 text-gray-200 hover:bg-black/90"
                }`}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 5c-7 0-10 7-10 7s3 7 10 7 10-7 10-7-3-7-10-7z" />
                </svg>
                {showHeatmap ? "Hide AI heatmap" : "Show AI heatmap"}
              </button>
            )}

          </div>
        )}

        {/* DETECT BUTTON */}

        <button
          onClick={handlePrediction}
          disabled={loading || !selectedImage}
          className="
          group mt-6 w-full
          inline-flex items-center justify-center gap-2.5
          rounded-2xl py-4 px-6
          text-base font-semibold text-white
          bg-gradient-to-br from-emerald-500 to-teal-600
          shadow-[0_8px_30px_-8px_rgba(16,185,129,0.7)]
          hover:from-emerald-400 hover:to-teal-500
          hover:shadow-[0_10px_36px_-6px_rgba(16,185,129,0.8)]
          disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none
          transition-all duration-300
          "
        >
          {loading ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
              </svg>
              Analysing the note…
            </>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16z" />
                <path d="m21 21-4.3-4.3" />
              </svg>
              Detect Currency
            </>
          )}
        </button>

        {/* ================================================= */}
        {/* RESULT */}
        {/* ================================================= */}

        {result && (

          <div className="mt-8 animate-in">

            {/* ================================================= */}
            {/* VERDICT BANNER — plain-language hero (Phase K) */}
            {/* ================================================= */}

            <VerdictBanner result={result} lang={lang} />

            {/* Report download (Phase N) */}
            <div className="mt-4 flex justify-end">
              <button
                onClick={downloadReport}
                className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-gray-200 hover:bg-white/10 transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download report (PDF)
              </button>
            </div>

            {/* ================================================= */}
            {/* WHAT WE FOUND — grouped plain findings (Phase K) */}
            {/* ================================================= */}

            <PlainFindings result={result} />

            {/* ================================================= */}
            {/* EXPLAIN WITH AI (Phase I) — plain explanation */}
            {/* ================================================= */}

            <ExplainPanel result={result} lang={lang} />

            {/* ================================================= */}
            {/* TECHNICAL DETAILS (collapsible) — for power users */}
            {/* ================================================= */}

            <details className="mt-8 group">
              <summary className="
              flex items-center gap-2
              cursor-pointer select-none
              text-sm font-semibold text-gray-300
              rounded-xl border border-white/10 bg-white/[0.03]
              px-4 py-3
              hover:bg-white/[0.06] hover:text-white
              transition-colors
              ">
                <svg className="h-4 w-4 transition-transform group-open:rotate-90" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="9 18 15 12 9 6" />
                </svg>
                Technical details — forensic checks, ML models, raw numbers
              </summary>

            {/* MODEL BREAKDOWN */}

            <div className="
            mt-6
            grid
            grid-cols-1
            md:grid-cols-2
            gap-6
            ">

              <div className="
              bg-white/[0.04]
              p-4
              rounded-2xl
              border
              border-white/10
              ">

                <h4 className="text-gray-400 text-sm mb-1">
                  ML Model Verdict
                </h4>

                <p
                  className={`
                  text-xl
                  font-bold
                  ${VERDICT_COLOR[result.model_verdict ?? ""]
                    ?? "text-gray-300"}
                  `}
                >
                  {result.model_verdict}
                  <span className="
                  text-gray-500
                  text-sm
                  font-normal
                  ml-2
                  ">
                    ({result.model_confidence})
                  </span>
                </p>

              </div>

              <div className="
              bg-white/[0.04]
              p-4
              rounded-2xl
              border
              border-white/10
              ">

                <h4 className="text-gray-400 text-sm mb-1">
                  Forensic Score
                </h4>

                <p className="text-xl font-bold text-blue-400">
                  {result.forensic_score?.toFixed(1)}%
                </p>

              </div>

            </div>

            {/* ================================================= */}
            {/* ML TECHNIQUE COMPARISON (Phase D) */}
            {/* ================================================= */}

            <ModelComparisonPanel models={result?.ml_models} />

            {/* ================================================= */}
            {/* OCR SECTION */}
            {/* ================================================= */}

            <div className="
            mt-8
            bg-white/[0.04]
            p-5
            rounded-2xl
            border
            border-white/10
            ">

              <h3 className="
              text-xl
              font-bold
              mb-3
              ">

                OCR Serial Number

              </h3>

              <p className="
              text-green-400
              break-words
              text-2xl
              font-mono
              ">

                {
                  (() => {
                    const v =
                      result?.forensic_analysis
                        ?.ocr_serial_number?.value;
                    return typeof v === "string" && v
                      ? v
                      : "Not Detected";
                  })()
                }

              </p>

              <p className="
              text-gray-500
              text-sm
              mt-2
              ">

                {
                  result?.forensic_analysis
                    ?.ocr_serial_number?.details
                  || ""
                }

              </p>

            </div>

            {/* ================================================= */}
            {/* PROPORTION ANALYSIS (dedicated panel) */}
            {/* ================================================= */}

            <ProportionPanel
              check={result?.forensic_analysis?.proportion_analysis}
            />

            {/* ================================================= */}
            {/* FORENSIC ANALYSIS */}
            {/* ================================================= */}

            <div className="mt-8">

              <h3 className="
              text-2xl
              font-bold
              mb-4
              ">

                Forensic Analysis

              </h3>

              <div className="
              grid
              grid-cols-1
              md:grid-cols-2
              gap-4
              ">


                <FeatureCard
                  title="Structural Sanity"
                  check={result?.forensic_analysis?.structural_sanity}
                />

                <FeatureCard
                  title="UV Light Detection"
                  check={result?.forensic_analysis?.uv_light_detection}
                />

                <FeatureCard
                  title="Watermark Detection"
                  check={result?.forensic_analysis?.watermark_detection}
                />

                <FeatureCard
                  title="OCR Serial Number"
                  check={result?.forensic_analysis?.ocr_serial_number}
                />

                <FeatureCard
                  title="Gandhi Face Analysis"
                  check={result?.forensic_analysis?.gandhi_face_analysis}
                />

                <FeatureCard
                  title="Security Thread Detection"
                  check={result?.forensic_analysis?.security_thread_detection}
                />
                {(() => {
                  const typo = result?.forensic_analysis?.serial_typography_analysis;
                  const v = typo?.value as any;
                  const status = typo?.status;
                  const sizes: number[] = v?.digit_sizes ?? [];
                  const pcts: number[] = v?.growth_percentages ?? [];
                  const rbiMatch: boolean | undefined = v?.rbi_match;

                  const statusBadge =
                    status === "PASS"
                      ? "bg-green-500/20 text-green-400 border-green-500/30"
                      : status === "FAIL"
                        ? "bg-red-500/20 text-red-400 border-red-500/30"
                        : "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";

                  return (
                    <div className="border border-white/10 rounded-2xl p-5 bg-white/[0.04] space-y-5">

                      {/* ---- Header + status badge ---- */}
                      <div className="flex items-center justify-between">
                        <h4 className="text-lg font-bold text-white">
                          RBI Serial Typography
                        </h4>
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${statusBadge}`}>
                          {status}
                        </span>
                      </div>

                      {/* ---- Simple explanation ---- */}
                      <p className="text-sm text-zinc-400 leading-relaxed">
                        {typo?.details}
                      </p>

                      {/* ---- Digit sizes with arrows ---- */}
                      {sizes.length > 0 && (
                        <div>
                          <p className="text-xs text-zinc-500 mb-2 uppercase tracking-wider">
                            Digit Sizes
                          </p>
                          <div className="flex items-center gap-1 flex-wrap">
                            {sizes.map((s, i) => (
                              <span key={i} className="flex items-center gap-1">
                                <span className="px-3 py-1.5 rounded-lg bg-white/[0.07] text-white font-mono text-sm font-semibold">
                                  {s}px
                                </span>
                                {i < sizes.length - 1 && (
                                  <span className="text-zinc-500 text-lg">→</span>
                                )}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* ---- Growth percentages as pills ---- */}
                      {pcts.length > 0 && (
                        <div>
                          <p className="text-xs text-zinc-500 mb-2 uppercase tracking-wider">
                            Growth per Step
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {pcts.map((g, i) => (
                              <span
                                key={i}
                                className={`px-3 py-1.5 rounded-full text-sm font-semibold ${
                                  g >= 0
                                    ? "bg-green-500/15 text-green-400"
                                    : "bg-red-500/15 text-red-400"
                                }`}
                              >
                                {g >= 0 ? "+" : ""}{g}%
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* ---- Total growth + RBI match row ---- */}
                      <div className="grid grid-cols-2 gap-3">

                        {v?.total_growth != null && (
                          <div className="bg-white/[0.07] rounded-xl p-3 text-center">
                            <p className="text-xs text-zinc-500 mb-1">Total Growth</p>
                            <p className="text-2xl font-bold text-cyan-400">
                              {v.total_growth}
                            </p>
                          </div>
                        )}

                        {rbiMatch !== undefined && (
                          <div className="bg-white/[0.07] rounded-xl p-3 text-center">
                            <p className="text-xs text-zinc-500 mb-1">RBI Pattern</p>
                            <span className={`inline-block px-4 py-1 rounded-full text-sm font-bold ${
                              rbiMatch
                                ? "bg-green-500/20 text-green-400"
                                : "bg-red-500/20 text-red-400"
                            }`}>
                              {rbiMatch ? "YES" : "NO"}
                            </span>
                          </div>
                        )}

                      </div>

                      {/* ---- Verdict bar ---- */}
                      {rbiMatch !== undefined && (
                        <div className={`rounded-xl p-3 text-center text-sm font-medium ${
                          rbiMatch
                            ? "bg-green-500/10 text-green-400 border border-green-500/20"
                            : "bg-red-500/10 text-red-400 border border-red-500/20"
                        }`}>
                          {rbiMatch
                            ? "Matches RBI increasing serial pattern"
                            : "Does not match RBI increasing serial pattern"}
                        </div>
                      )}

                    </div>
                  );
                })()}

                <FeatureCard
                  title="Micro-lettering / Fine Print"
                  check={result?.forensic_analysis?.microprint_detection}
                />

                <FeatureCard
                  title="Colour Palette Integrity"
                  check={result?.forensic_analysis?.hologram_detection}
                />

                <FeatureCard
                  title="Denomination Classification"
                  check={result?.forensic_analysis?.denomination_classification}
                />

                <FeatureCard
                  title="Proportion Analysis"
                  check={result?.forensic_analysis?.proportion_analysis}
                />

                <FeatureCard
                  title="Bleed Lines (edge)"
                  check={result?.forensic_analysis?.bleed_line_detection}
                />

                <FeatureCard
                  title="Identification Mark (tactile)"
                  check={result?.forensic_analysis?.identification_mark}
                />

                <FeatureCard
                  title="Digital Tamper (ELA)"
                  check={result?.forensic_analysis?.tamper_detection}
                />

                <FeatureCard
                  title="Modular AI Pipeline"
                  check={result?.forensic_analysis?.modular_ai_pipeline}
                />

              </div>
            </div>
            </details>
          </div>
        )}
      </div>

      {/* ================================================= */}
      {/* SECURITY PATTERN STUDIO (Phase J) */}
      {/* ================================================= */}

      <SecurityPatternStudio />

      {/* ================================================= */}
      {/* LIVE CAMERA CAPTURE (Phase O) */}
      {/* ================================================= */}

      {cameraOpen && (
        <CameraModal
          onClose={() => setCameraOpen(false)}
          onCapture={(file) => { acceptFile(file); setCameraOpen(false); }}
        />
      )}

      {/* ================================================= */}
      {/* HELP CHATBOT (Phase L) — floating assistant */}
      {/* ================================================= */}

      <ChatAssistant />

    </main>
  );
}

// =====================================================
// FEATURE CARD COMPONENT
// =====================================================

const STATUS_STYLES = {
  PASS: {
    badge: "bg-green-500/20 text-green-400 border-green-500/40",
    border: "border-green-500/40",
    label: "PASS",
  },
  FAIL: {
    badge: "bg-red-500/20 text-red-400 border-red-500/40",
    border: "border-red-500/40",
    label: "FAIL",
  },
  INFO: {
    badge: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
    border: "border-yellow-500/40",
    label: "INFO",
  },
} as const;

function FeatureCard({
  title,
  check,
}: {
  title: string;
  check?: ForensicCheck;
}) {

  const status = check?.status ?? "INFO";
  const style = STATUS_STYLES[status];

  return (

    <div
      className={`
      bg-white/[0.04]
      p-4
      rounded-xl
      border
      ${style.border}
      transition-all
      duration-300
      `}
    >

      <div className="
      flex
      justify-between
      items-center
      mb-2
      ">

        <span className="font-semibold">{title}</span>

        <span
          className={`
          text-xs
          font-bold
          px-2
          py-0.5
          rounded-full
          border
          ${style.badge}
          `}
        >
          {style.label}
        </span>

      </div>

      <p className="text-sm text-gray-400 break-words">
        {check?.details ?? "Awaiting result"}
      </p>

      {typeof check?.value === "string" && (
        <p className="
        text-sm
        text-green-400
        font-mono
        mt-1
        break-words
        ">
          {check.value}
        </p>
      )}

    </div>
  );
}

// =====================================================
// PROPORTION ANALYSIS PANEL (Phase C-1)
// =====================================================
// Banknote proportion check surfaces the measured note quad
// aspect vs the canonical RBI aspect for the OCR'd
// denomination and renders the deviation prominently. A
// fake printed on wrong-size paper or a digitally stretched
// real-note image lands here.

function ProportionPanel({ check }: { check?: ForensicCheck }) {

  if (!check) return null;

  const status = check.status;
  const style = STATUS_STYLES[status];
  const v = isProportionValue(check.value) ? check.value : null;

  const deviationColor =
    !v
      ? "text-gray-400"
      : v.deviation_pct <= 5
        ? "text-green-400"
        : v.deviation_pct <= 15
          ? "text-yellow-400"
          : "text-red-400";

  return (

    <div className={`
    mt-8
    bg-white/[0.04]
    p-5
    rounded-2xl
    border
    ${style.border}
    `}>

      <div className="
      flex
      justify-between
      items-center
      mb-3
      ">

        <h3 className="text-xl font-bold">
          Proportion Analysis
        </h3>

        <span className={`
        text-xs
        font-bold
        px-2
        py-0.5
        rounded-full
        border
        ${style.badge}
        `}>
          {style.label}
        </span>

      </div>

      {v ? (

        <div className="
        grid
        grid-cols-3
        gap-4
        mb-3
        ">

          <div>
            <p className="text-xs text-gray-500 mb-1">
              Measured Aspect
            </p>
            <p className="text-2xl font-mono text-white">
              {v.actual_aspect.toFixed(3)}
            </p>
            <p className="text-xs text-gray-600 mt-1">
              via {v.measurement === "quad"
                ? "detected note edges"
                : "image frame"}
            </p>
          </div>

          <div>
            <p className="text-xs text-gray-500 mb-1">
              RBI Canonical
            </p>
            <p className="text-2xl font-mono text-white">
              {v.expected_aspect.toFixed(3)}
            </p>
            <p className="text-xs text-gray-600 mt-1">
              for this denomination
            </p>
          </div>

          <div>
            <p className="text-xs text-gray-500 mb-1">
              Deviation
            </p>
            <p className={`
            text-2xl
            font-mono
            font-bold
            ${deviationColor}
            `}>
              {v.deviation_pct.toFixed(1)}%
            </p>
            <p className="text-xs text-gray-600 mt-1">
              tolerance 15.0%
            </p>
          </div>

        </div>

      ) : null}

      <p className="text-sm text-gray-400 break-words">
        {check.details}
      </p>

    </div>
  );
}

// =====================================================
// ML TECHNIQUE COMPARISON PANEL (Phase D)
// =====================================================
// Surfaces the two machine-learning views side by side:
//   - MobileNetV2 (CNN) on raw pixels
//   - the best classical technique (Phase D benchmark) on
//     hand-crafted visual features, as an independent second
//     opinion.
// Display-only: the classical model does not drive the combined
// verdict (fusion recalibration is Phase F). An agreement badge
// makes model disagreement visible at a glance.

const PRETTY_CLASSICAL: Record<string, string> = {
  random_forest: "Random Forest",
  svm_rbf: "SVM (RBF)",
  logistic_regression: "Logistic Regression",
  knn: "KNN",
};

function ModelVerdictTile({
  title,
  verdict,
  confidence,
  subtitle,
}: {
  title: string;
  verdict?: string | null;
  confidence?: string | null;
  subtitle: string;
}) {
  return (
    <div className="bg-white/[0.04] p-4 rounded-2xl border border-white/10">
      <h4 className="text-gray-400 text-sm mb-1">{title}</h4>
      <p
        className={`text-xl font-bold ${
          VERDICT_COLOR[verdict ?? ""] ?? "text-gray-300"
        }`}
      >
        {verdict ?? "N/A"}
        {confidence && (
          <span className="text-gray-500 text-sm font-normal ml-2">
            ({confidence})
          </span>
        )}
      </p>
      <p className="text-xs text-gray-600 mt-1">{subtitle}</p>
    </div>
  );
}

function ModelComparisonPanel({ models }: { models?: MlModels }) {

  if (!models) return null;

  const classical = models.classical;
  const classicalName =
    classical?.name && PRETTY_CLASSICAL[classical.name]
      ? PRETTY_CLASSICAL[classical.name]
      : classical?.name ?? "Classical model";

  const agreement = models.agreement;

  return (
    <div className="mt-8 bg-white/[0.04] p-5 rounded-2xl border border-white/10">

      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">ML Technique Comparison</h3>
        {agreement !== null && agreement !== undefined && (
          <span
            className={`text-xs font-bold px-3 py-1 rounded-full border ${
              agreement
                ? "bg-green-500/20 text-green-400 border-green-500/40"
                : "bg-yellow-500/20 text-yellow-400 border-yellow-500/40"
            }`}
          >
            {agreement ? "MODELS AGREE" : "MODELS DISAGREE"}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        <ModelVerdictTile
          title="MobileNetV2 (CNN)"
          verdict={models.cnn?.verdict}
          confidence={models.cnn?.confidence}
          subtitle="Deep model on raw pixels"
        />

        {classical?.available ? (
          <ModelVerdictTile
            title={classicalName}
            verdict={classical.verdict}
            confidence={classical.confidence}
            subtitle="Classical model on visual features (2nd opinion)"
          />
        ) : (
          <div className="bg-white/[0.04] p-4 rounded-2xl border border-white/10">
            <h4 className="text-gray-400 text-sm mb-1">Classical model</h4>
            <p className="text-xl font-bold text-gray-500">Not trained</p>
            <p className="text-xs text-gray-600 mt-1">
              Run scripts/train_classical.py to enable
            </p>
          </div>
        )}

      </div>

      <p className="text-xs text-gray-600 mt-3">
        Second opinion shown for transparency — does not change the
        combined verdict above.
      </p>

    </div>
  );
}

// =====================================================
// EXPLAIN WITH AI PANEL (Phase I)
// =====================================================
// Turns the structured verdict into a plain-language, accessibility-first
// explanation (the use case both reference papers targeted) plus manual
// verification steps. Calls POST /explain, which uses Claude when an API key
// is configured and otherwise returns a deterministic template — so the panel
// always works locally. This is the project's "GenAI that does something good":
// explainability, NOT counterfeit generation.

function ExplainPanel({ result, lang }: { result: PredictResponse; lang: Lang }) {

  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [source, setSource] = useState<"llm" | "template" | null>(null);
  const [error, setError] = useState<string>("");
  const [speaking, setSpeaking] = useState(false);

  // If the language changes, the existing explanation is stale — clear it.
  useEffect(() => {
    setExplanation(null);
    setSource(null);
    stopSpeaking();
    setSpeaking(false);
  }, [lang]);

  const handleExplain = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.post<ExplainResponse>(
        "http://127.0.0.1:8000/explain",
        { ...result, lang },
        { headers: { "Content-Type": "application/json" } }
      );

      if (response.data.status === "success" && response.data.explanation) {
        setExplanation(response.data.explanation);
        setSource(response.data.explanation.source);
      } else {
        setError(response.data.message ?? "Could not generate an explanation.");
      }
    } catch {
      setError("Backend connection failed");
    } finally {
      setLoading(false);
    }
  };

  const toListen = () => {
    if (!explanation) return;
    if (speaking) { stopSpeaking(); setSpeaking(false); return; }
    const text = [
      explanation.summary,
      ...explanation.reasons,
      ...explanation.manual_checks,
    ].join(". ");
    speak(text, lang);
    setSpeaking(true);
  };

  return (
    <div className="mt-8 bg-white/[0.04] p-5 rounded-2xl border border-white/10">

      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">{t("explainTitle", lang)}</h3>
        {source && (
          <span
            className={`text-xs font-bold px-3 py-1 rounded-full border ${
              source === "llm"
                ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                : "bg-white/[0.07] text-gray-300 border-white/15"
            }`}
          >
            {source === "llm" ? "AI GENERATED" : "RULE-BASED"}
          </span>
        )}
      </div>

      {!explanation && (
        <p className="text-sm text-gray-400 mb-4">
          Get a plain-language summary of why this verdict was reached, plus
          manual checks you can do by hand. Designed to be read aloud.
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        <button
          onClick={handleExplain}
          disabled={loading}
          className="
          bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60
          transition-all duration-300
          px-5 py-2.5 rounded-xl text-sm font-semibold
          "
        >
          {loading
            ? t("generating", lang)
            : explanation
              ? t("regenerate", lang)
              : t("explainCta", lang)}
        </button>

        {explanation && (
          <button
            onClick={toListen}
            className="flex items-center gap-1.5 rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-semibold text-gray-200 hover:bg-white/10 transition-colors"
          >
            {speaking ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14" />
              </svg>
            )}
            {speaking ? t("stop", lang) : t("listen", lang)}
          </button>
        )}
      </div>

      {error && (
        <p className="text-sm text-red-400 mt-3">{error}</p>
      )}

      {explanation && (
        <div className="mt-5 space-y-5">

          {/* ---- Summary ---- */}
          <div>
            <p className="text-xs text-gray-500 mb-1 uppercase tracking-wider">
              {t("summary", lang)}
            </p>
            <p className="text-base text-gray-200 leading-relaxed">
              {explanation.summary}
            </p>
          </div>

          {/* ---- Reasons ---- */}
          {explanation.reasons.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">
                {t("why", lang)}
              </p>
              <ul className="space-y-1.5">
                {explanation.reasons.map((r, i) => (
                  <li
                    key={i}
                    className="text-sm text-gray-300 flex gap-2 leading-relaxed"
                  >
                    <span className="text-indigo-400 mt-0.5">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* ---- Manual checks ---- */}
          {explanation.manual_checks.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">
                {t("byHand", lang)}
              </p>
              <ul className="space-y-1.5">
                {explanation.manual_checks.map((m, i) => (
                  <li
                    key={i}
                    className="text-sm text-gray-300 flex gap-2 leading-relaxed"
                  >
                    <span className="text-green-400 mt-0.5">✓</span>
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="text-xs text-gray-600">
            Screening aid only — software on a phone photo can be wrong,
            especially on high-quality fakes. Always verify by hand when it
            matters.
          </p>

        </div>
      )}

    </div>
  );
}

// =====================================================
// SECURITY PATTERN STUDIO (Phase J)
// =====================================================
// The "generative" half of the project, done legitimately. Real currency
// resists cloning through physics + the deterministic mathematical complexity
// of guilloché engraving. This studio procedurally generates that kind of
// ABSTRACT ornament (woven sinusoidal lattices + spirograph rosettes +
// micro-text) from a seed. Same seed → same art, so a pattern keyed to a serial
// can be regenerated and visually compared. It is decorative art — it does NOT
// generate currency.

const BACKEND = "http://127.0.0.1:8000";

function SecurityPatternStudio() {

  const [seedInput, setSeedInput] = useState("ABHI-500-2026");
  const [appliedSeed, setAppliedSeed] = useState("ABHI-500-2026");

  const generate = () => {
    const s = seedInput.trim() || "0";
    setAppliedSeed(s);
  };

  const imgUrl =
    `${BACKEND}/security-pattern?seed=${encodeURIComponent(appliedSeed)}&size=600`;

  return (
    <div className="
    mt-10
    bg-white/[0.04]
    border
    border-white/10
    rounded-3xl
    shadow-2xl
    p-8
    w-full
    max-w-4xl
    ">

      <h2 className="text-3xl font-bold mb-2">
        Security Pattern Studio
      </h2>

      <p className="text-gray-400 mb-6 text-sm max-w-2xl">
        Procedurally generated <span className="text-gray-200">guilloché</span> art
        — the kind of interwoven mathematical engraving used on real security
        documents. It is fully deterministic: the same seed always produces the
        same pattern, so a design keyed to a serial number can be regenerated and
        compared. Abstract ornament only — not currency.
      </p>

      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <input
          type="text"
          value={seedInput}
          onChange={(e) => setSeedInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") generate(); }}
          placeholder="Seed (e.g. a serial number)"
          className="
          flex-1
          bg-black
          border
          border-white/10
          rounded-xl
          px-4
          py-3
          text-sm
          text-gray-200
          font-mono
          focus:outline-none
          focus:border-indigo-500
          "
        />
        <button
          onClick={generate}
          className="
          bg-indigo-600
          hover:bg-indigo-500
          transition-all
          duration-300
          px-6
          py-3
          rounded-xl
          text-sm
          font-semibold
          whitespace-nowrap
          "
        >
          Generate Pattern
        </button>
      </div>

      <div className="flex justify-center">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imgUrl}
          alt={`Generated guilloché security pattern for seed ${appliedSeed}`}
          className="
          rounded-2xl
          border
          border-white/10
          bg-white
          w-full
          max-w-md
          aspect-square
          object-contain
          "
        />
      </div>

      <p className="text-xs text-gray-600 mt-4 text-center font-mono">
        seed: {appliedSeed}
      </p>

    </div>
  );
}

// =====================================================
// VERDICT BANNER (Phase K)
// =====================================================
// The plain-language hero of the result. A normal user reads this and
// knows what to do — without parsing 11 technical checks. Handles the
// honest "Can't verify — retake" state and the "limited check" caveat.

const RETAKE_TIPS = [
  "Fill the frame with the note (get close).",
  "Use good, even light — avoid glare and shadows.",
  "Lay the note flat and hold the camera straight above it.",
  "Make sure both serial numbers are sharp and in focus.",
];

function VerdictBanner({ result, lang }: { result: PredictResponse; lang: Lang }) {

  const verdict = result.prediction ?? "SUSPICIOUS";
  const display = VERDICT_DISPLAY[verdict] ?? VERDICT_DISPLAY.SUSPICIOUS;
  const level = result.verification_level;

  // Hindi headline/sub when selected (colours come from the English map).
  const hi = VERDICT_HEADLINE_HI[verdict];
  const headline = lang === "hi" && hi ? hi.headline : display.headline;
  const sub = lang === "hi" && hi ? hi.sub : display.sub;

  const denomRaw = result.forensic_analysis?.denomination_classification?.value;
  const denom = typeof denomRaw === "string" ? denomRaw : null;

  const isUnverified = verdict === "UNVERIFIED";
  const isLimited = !isUnverified && level === "partial";

  const [speaking, setSpeaking] = useState(false);

  const toListen = () => {
    if (speaking) { stopSpeaking(); setSpeaking(false); return; }
    const parts = [headline + (denom && !isUnverified ? ` ₹${denom}` : ""), sub];
    if (result.guidance) parts.push(result.guidance);
    speak(parts.join(". "), lang);
    setSpeaking(true);
  };

  return (
    <div className={`rounded-2xl border ${display.border} ${display.bg} p-6`}>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-400 mb-1">
            {t("result", lang)}
          </p>
          <h2 className={`text-4xl font-bold ${display.accent}`}>
            {headline}
            {denom && !isUnverified && (
              <span className="text-gray-200 text-2xl font-semibold ml-2">
                ₹{denom}
              </span>
            )}
          </h2>
        </div>

        <div className="flex items-center gap-4">
          {!isUnverified && (
            <div className="text-right">
              <p className="text-xs text-gray-400">{t("confidence", lang)}</p>
              <p className={`text-2xl font-bold ${display.accent}`}>
                {result.confidence}
              </p>
            </div>
          )}
          <button
            onClick={toListen}
            aria-label={speaking ? t("stop", lang) : t("listen", lang)}
            className="flex items-center gap-1.5 rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm text-gray-200 hover:bg-white/10 transition-colors"
          >
            {speaking ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14" />
              </svg>
            )}
            {speaking ? t("stop", lang) : t("listen", lang)}
          </button>
        </div>
      </div>

      <p className="text-sm text-gray-300 mt-3 leading-relaxed">
        {sub}
      </p>

      {/* Limited-check caveat (note read only partially) */}
      {isLimited && result.guidance && (
        <div className="mt-4 rounded-xl bg-black/30 border border-yellow-500/30 p-3">
          <p className="text-sm text-yellow-300">
            ⚠ {result.guidance}
          </p>
        </div>
      )}

      {/* Retake guidance (couldn't read the note at all) */}
      {isUnverified && (
        <div className="mt-4 rounded-xl bg-black/30 border border-orange-500/30 p-4">
          <p className="text-sm text-orange-200 font-semibold mb-2">
            {t("retakeHow", lang)}
          </p>
          <ul className="space-y-1.5">
            {RETAKE_TIPS.map((tip, i) => (
              <li key={i} className="text-sm text-gray-300 flex gap-2">
                <span className="text-orange-400 mt-0.5">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

    </div>
  );
}

// =====================================================
// PLAIN FINDINGS (Phase K)
// =====================================================
// Re-expresses the technical checks as a few human categories with plain
// English and a clear ✓ / ⚠ / — status, so a non-technical user can
// "seriously understand" what was and wasn't checked. The raw numbers
// still live in the Technical details expander below.

type PlainCheck = {
  key: keyof ForensicAnalysis;
  group: string;
  label: string;
  good: string;
  fail: string;
  info: string;
};

const PLAIN_CHECKS: PlainCheck[] = [
  { key: "ocr_serial_number", group: "Identity", label: "Serial number",
    good: "Read the note's serial number.",
    fail: "Couldn't read a serial number.",
    info: "Serial number wasn't readable in this photo." },
  { key: "denomination_classification", group: "Identity", label: "Denomination",
    good: "Recognised the note's value.",
    fail: "Couldn't recognise the value.",
    info: "Value wasn't clearly readable." },
  { key: "proportion_analysis", group: "Size & shape", label: "Size & shape",
    good: "Dimensions match a real note of this value.",
    fail: "Dimensions look wrong — possible stretch or wrong-size paper.",
    info: "Couldn't measure the note's size in this photo." },
  { key: "watermark_detection", group: "Security features", label: "Watermark",
    good: "The watermark area looks right.",
    fail: "The watermark area looks wrong.",
    info: "Couldn't assess the watermark." },
  { key: "security_thread_detection", group: "Security features", label: "Security thread",
    good: "Found the vertical security thread.",
    fail: "Couldn't find the security thread.",
    info: "Security thread wasn't assessed." },
  { key: "uv_light_detection", group: "Security features", label: "Special ink",
    good: "Ink response looks consistent with a real note.",
    fail: "Ink response looks off.",
    info: "Proper check needs UV light — shown for information only." },
  { key: "microprint_detection", group: "Security features", label: "Tiny print",
    good: "Fine micro-print looks intact.",
    fail: "Fine micro-print looks lost.",
    info: "Photo isn't sharp enough to check the tiny print." },
  { key: "bleed_line_detection", group: "Security features", label: "Bleed lines",
    good: "The edge bleed-line count matches a real note of this value.",
    fail: "The edge bleed-line count looks wrong.",
    info: "Couldn't count the edge bleed lines at this resolution." },
  { key: "identification_mark", group: "Security features", label: "ID mark (touch)",
    good: "A raised identification mark is present where it should be.",
    fail: "The identification mark looks wrong.",
    info: "Couldn't confirm the raised identification mark — it's a touch feature." },
  { key: "tamper_detection", group: "Look & feel", label: "Digital editing",
    good: "No obvious digital editing of the image.",
    fail: "Signs of digital editing in the image.",
    info: "Image-tamper (ELA) reading — see details; informational only." },
  { key: "structural_sanity", group: "Look & feel", label: "Overall look",
    good: "Overall structure looks like a banknote.",
    fail: "Doesn't look like a proper banknote.",
    info: "Overall structure was unclear." },
  { key: "hologram_detection", group: "Look & feel", label: "Colours",
    good: "Colours look right for a real note.",
    fail: "Colours look wrong — washed out or off.",
    info: "Colours weren't assessed." },
  { key: "gandhi_face_analysis", group: "Look & feel", label: "Portrait",
    good: "Found the Gandhi portrait.",
    fail: "The portrait looks wrong.",
    info: "Couldn't auto-find the portrait (this detector is unreliable, so it's not counted)." },
];

const GROUP_ORDER = ["Identity", "Size & shape", "Security features", "Look & feel"];

function statusToPlain(status?: "PASS" | "FAIL" | "INFO") {
  if (status === "PASS")
    return { icon: "✓", color: "text-green-400", state: "good" as const };
  if (status === "FAIL")
    return { icon: "✗", color: "text-red-400", state: "bad" as const };
  return { icon: "—", color: "text-gray-500", state: "unknown" as const };
}

function PlainFindings({ result }: { result: PredictResponse }) {

  const fa = result.forensic_analysis;
  if (!fa) return null;

  return (
    <div className="mt-8 bg-white/[0.04] border border-white/10 rounded-2xl p-5">

      <h3 className="text-xl font-bold mb-1">What we checked</h3>
      <p className="text-sm text-gray-500 mb-5">
        Plain-language summary. ✓ looks right · ✗ a problem · — couldn&apos;t check.
      </p>

      <div className="space-y-5">
        {GROUP_ORDER.map((group) => {
          const items = PLAIN_CHECKS.filter((c) => c.group === group);
          return (
            <div key={group}>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
                {group}
              </p>
              <div className="space-y-2">
                {items.map((c) => {
                  const check = fa[c.key];
                  const st = statusToPlain(check?.status);
                  const text =
                    st.state === "good" ? c.good
                      : st.state === "bad" ? c.fail
                        : c.info;
                  return (
                    <div key={c.key} className="flex gap-3 items-start">
                      <span className={`text-lg leading-6 ${st.color}`}>
                        {st.icon}
                      </span>
                      <div>
                        <span className="text-sm font-semibold text-gray-200">
                          {c.label}
                        </span>
                        <span className="text-sm text-gray-400"> — {text}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

// =====================================================
// HELP CHATBOT (Phase L)
// =====================================================
// A floating assistant that answers "how does this work / how do I run it /
// what does this verdict mean". Talks to POST /chat, which uses Claude when an
// API key is set and a deterministic FAQ otherwise — so it works in a live
// demo with no internet.

type ChatMsg = { role: "user" | "assistant"; content: string };

const CHAT_GREETING: ChatMsg = {
  role: "assistant",
  content:
    "Hi! I'm the help assistant for this project. Ask me how it works, how to " +
    "run it, what a verdict means, or what the Security Pattern Studio is.",
};

const CHAT_SUGGESTIONS = [
  "How does it work?",
  "What does UNVERIFIED mean?",
  "How accurate is it?",
];

function ChatAssistant() {

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>([CHAT_GREETING]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, open]);

  const send = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || loading) return;

    const history = messages;
    setMessages((m) => [...m, { role: "user", content: message }]);
    setInput("");
    setLoading(true);

    try {
      const r = await axios.post(`${BACKEND}/chat`, { message, history });
      const reply =
        r.data?.status === "success" && r.data.reply
          ? r.data.reply
          : "Sorry, I couldn't answer that. Is the backend running on port 8000?";
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "I couldn't reach the backend. Make sure it's running " +
            "(uvicorn backend.main:app on port 8000).",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating toggle button */}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close help assistant" : "Open help assistant"}
        className="
        fixed bottom-6 right-6 z-50
        h-14 w-14 rounded-2xl
        bg-gradient-to-br from-indigo-500 to-violet-600
        hover:from-indigo-400 hover:to-violet-500
        shadow-[0_10px_30px_-6px_rgba(99,102,241,0.7)]
        flex items-center justify-center text-white
        transition-all duration-300 hover:scale-105 active:scale-95
        "
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
        )}
      </button>

      {/* Chat panel */}
      {open && (
        <div className="
        fixed bottom-24 right-6 z-50
        w-[92vw] max-w-sm h-[30rem]
        rounded-3xl border border-white/12
        bg-[#0c0e16]/95 backdrop-blur-2xl
        shadow-[0_24px_70px_-12px_rgba(0,0,0,0.8)]
        flex flex-col overflow-hidden animate-in
        ">

          <div className="px-4 py-3.5 border-b border-white/10 bg-gradient-to-r from-indigo-500/15 to-violet-500/10 flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-lg">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 8V4H8" /><rect x="4" y="8" width="16" height="12" rx="2" />
                <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
              </svg>
            </span>
            <div>
              <p className="font-semibold text-sm leading-tight">Help Assistant</p>
              <p className="text-xs text-gray-400">How it works · how to run it</p>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`
                  max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap
                  ${m.role === "user"
                    ? "bg-indigo-600 text-white rounded-br-sm"
                    : "bg-white/[0.07] text-gray-200 rounded-bl-sm"}
                  `}
                >
                  {m.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white/[0.07] text-gray-400 rounded-2xl px-3 py-2 text-sm">
                  Thinking…
                </div>
              </div>
            )}

            {messages.length === 1 && (
              <div className="flex flex-wrap gap-2 pt-1">
                {CHAT_SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="text-xs px-3 py-1.5 rounded-full border border-white/15 text-gray-300 hover:bg-white/[0.07]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="p-3 border-t border-white/10 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") send(); }}
              placeholder="Ask a question…"
              className="
              flex-1 bg-black border border-white/10 rounded-xl
              px-3 py-2 text-sm text-gray-200
              focus:outline-none focus:border-indigo-500
              "
            />
            <button
              onClick={() => send()}
              disabled={loading || !input.trim()}
              aria-label="Send"
              className="
              flex items-center justify-center
              bg-gradient-to-br from-indigo-500 to-violet-600
              hover:from-indigo-400 hover:to-violet-500
              disabled:opacity-40 disabled:cursor-not-allowed
              px-3.5 rounded-xl text-white transition-all
              "
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>

        </div>
      )}
    </>
  );
}
// =====================================================
// LIVE CAMERA CAPTURE MODAL (Phase O)
// =====================================================
// Opens the device camera (rear-facing on mobile), shows a live preview, and
// captures a still frame as a JPEG File fed into the same /predict flow.
// Pure browser APIs (getUserMedia + canvas) — no new dependencies. Cleans up
// the media stream on close/unmount.

function CameraModal({
  onClose,
  onCapture,
}: {
  onClose: () => void;
  onCapture: (file: File) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const aliveRef = useRef(true);

  // phase: idle (need a click) → starting → live, or error
  const [phase, setPhase] = useState<"idle" | "starting" | "live" | "error">("idle");
  const [error, setError] = useState<string>("");

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  // Only cleanup on unmount — the camera is started by a user click (not on
  // mount), which avoids the React-StrictMode double-invoke that re-opens the
  // device too fast and trips a NotReadableError.
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      stopStream();
    };
  }, []);

  const start = async () => {
    setPhase("starting");
    setError("");
    stopStream();
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw Object.assign(new Error("no api"), { name: "NotSupported" });
      }
      // All "ideal" → soft preferences, so this never over-constrains and
      // falls back to whatever camera exists (front cam on a laptop).
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (!aliveRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setPhase("live");
    } catch (e) {
      const name = (e as { name?: string })?.name ?? "";
      let msg: string;
      if (name === "NotAllowedError" || name === "SecurityError") {
        msg =
          "Camera permission is blocked. Click the camera/🔒 icon at the left " +
          "of the address bar → set Camera to Allow → then press Try again.";
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        msg = "No camera was found on this device.";
      } else if (name === "NotReadableError" || name === "TrackStartError") {
        msg =
          "The camera is already in use by another app or browser tab. Close " +
          "it (Zoom/Meet/another tab) and press Try again.";
      } else if (name === "NotSupported") {
        msg = "This browser doesn't expose a camera API (needs https or localhost).";
      } else {
        msg = `Couldn't start the camera (${name || "unknown error"}). Press Try again.`;
      }
      if (aliveRef.current) {
        setError(msg);
        setPhase("error");
      }
    }
  };

  const capture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Crop to the central banknote-shaped framing box (must match the on-screen
    // guide below) so the captured image is mostly the NOTE, not the room —
    // this is what makes the denomination/serial actually readable.
    let cw = Math.round(vw * 0.86);
    let ch = Math.round(cw / 2.0);          // ~banknote aspect
    if (ch > vh * 0.9) {
      ch = Math.round(vh * 0.9);
      cw = Math.min(vw, Math.round(ch * 2.0));
    }
    const cx = Math.round((vw - cw) / 2);
    const cy = Math.round((vh - ch) / 2);

    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, cx, cy, cw, ch, 0, 0, cw, ch);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        stopStream();
        onCapture(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
      },
      "image/jpeg",
      0.95
    );
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="card w-full max-w-2xl p-5 animate-in">

        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">Capture a note</h3>
          <button
            onClick={onClose}
            aria-label="Close camera"
            className="rounded-lg p-1.5 text-gray-400 hover:bg-white/10 hover:text-white"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* ---- IDLE: ask permission on a click (Google-Meet style) ---- */}
        {phase === "idle" && (
          <div className="flex flex-col items-center text-center py-8">
            <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-400/30 mb-4">
              <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </span>
            <p className="text-sm text-gray-400 max-w-sm mb-5">
              We&apos;ll ask your browser for camera access. Choose
              <span className="text-emerald-300 font-medium"> Allow</span> when prompted.
            </p>
            <button
              onClick={start}
              className="inline-flex items-center justify-center gap-2 rounded-xl py-3 px-6 text-sm font-semibold text-white bg-gradient-to-br from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 transition-all"
            >
              Enable camera
            </button>
          </div>
        )}

        {/* ---- ERROR: message + retry ---- */}
        {phase === "error" && (
          <div className="flex flex-col items-center text-center py-8">
            <p className="text-sm text-red-400 max-w-md mb-5">{error}</p>
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="rounded-xl border border-white/15 bg-white/5 py-2.5 px-5 text-sm font-semibold text-gray-200 hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={start}
                className="rounded-xl py-2.5 px-6 text-sm font-semibold text-white bg-gradient-to-br from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 transition-all"
              >
                Try again
              </button>
            </div>
          </div>
        )}

        {/* ---- STARTING / LIVE ---- */}
        {(phase === "starting" || phase === "live") && (
          <>
            <div className="relative overflow-hidden rounded-2xl bg-black ring-1 ring-white/10">
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video ref={videoRef} playsInline muted autoPlay className="w-full h-auto block" />
              {phase === "starting" && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
                  Starting camera…
                </div>
              )}
              {/* Banknote-shaped guide — ONLY this box is captured. */}
              <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[86%] aspect-[2/1] max-h-[88%] rounded-xl border-2 border-dashed border-emerald-400/80 shadow-[0_0_0_9999px_rgba(0,0,0,0.35)]" />
            </div>

            <p className="text-xs text-gray-400 mt-3 text-center">
              Put the note <span className="text-emerald-300 font-medium">inside the green box and fill it</span> — only the box is captured. Hold steady in good light.
            </p>

            <div className="mt-4 flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 rounded-xl border border-white/15 bg-white/5 py-3 text-sm font-semibold text-gray-200 hover:bg-white/10 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={capture}
                disabled={phase !== "live"}
                className="flex-1 inline-flex items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold text-white bg-gradient-to-br from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 disabled:opacity-40 transition-all"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
                Capture
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
