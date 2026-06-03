"use client";

import { useState } from "react";
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

type Verdict = "REAL" | "FAKE" | "SUSPICIOUS" | "UNVERIFIED";

type PredictResponse = {
  status: "success" | "error";
  prediction?: Verdict;
  security_verdict?: "REAL" | "FAKE" | "SUSPICIOUS";
  verification_level?: "full" | "partial" | "none";
  verification?: Verification;
  guidance?: string;
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

  // =====================================================
  // HANDLE IMAGE CHANGE
  // =====================================================

  const handleImageChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    const file = e.target.files?.[0];

    if (file) {

      setSelectedImage(file);

      setPreview(
        URL.createObjectURL(file)
      );

      setResult(null);
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

    <main className="
    min-h-screen
    bg-black
    text-white
    flex
    flex-col
    items-center
    px-6
    py-10
    ">

      {/* ================================================= */}
      {/* TITLE */}
      {/* ================================================= */}

      <h1 className="
      text-5xl
      font-bold
      text-center
      mb-4
      ">

        AI Counterfeit Currency Detection

      </h1>

      <p className="
      text-gray-400
      text-center
      mb-10
      max-w-3xl
      ">

        Deep Learning based forensic currency
        authentication system using
        MobileNetV2 and modular AI pipeline
        architecture.

      </p>

      {/* ================================================= */}
      {/* MAIN CARD */}
      {/* ================================================= */}

      <div className="
      bg-zinc-900
      border
      border-zinc-800
      rounded-3xl
      shadow-2xl
      p-8
      w-full
      max-w-4xl
      ">

        {/* FILE INPUT */}

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="
          mb-6
          block
          w-full
          text-sm
          text-gray-300
          file:mr-4
          file:py-3
          file:px-6
          file:rounded-xl
          file:border-0
          file:text-sm
          file:font-semibold
          file:bg-green-500
          file:text-white
          hover:file:bg-green-600
          "
        />

        {/* IMAGE PREVIEW */}

        {preview && (

          <div className="mb-8">

            <img
              src={preview}
              alt="Preview"
              className="
              rounded-2xl
              w-full
              h-72
              object-contain
              border
              border-zinc-700
              bg-black
              "
            />

          </div>
        )}

        {/* BUTTON */}

        <button
          onClick={handlePrediction}
          disabled={loading}
          className="
          w-full
          bg-green-500
          hover:bg-green-600
          transition-all
          duration-300
          py-4
          rounded-2xl
          text-lg
          font-semibold
          "
        >

          {
            loading
              ? "Detecting Currency..."
              : "Detect Currency"
          }

        </button>

        {/* ================================================= */}
        {/* RESULT */}
        {/* ================================================= */}

        {result && (

          <div className="
          mt-10
          bg-black
          border
          border-zinc-800
          rounded-2xl
          p-6
          ">

            {/* ================================================= */}
            {/* VERDICT BANNER — plain-language hero (Phase K) */}
            {/* ================================================= */}

            <VerdictBanner result={result} />

            {/* ================================================= */}
            {/* WHAT WE FOUND — grouped plain findings (Phase K) */}
            {/* ================================================= */}

            <PlainFindings result={result} />

            {/* ================================================= */}
            {/* EXPLAIN WITH AI (Phase I) — plain explanation */}
            {/* ================================================= */}

            <ExplainPanel result={result} />

            {/* ================================================= */}
            {/* TECHNICAL DETAILS (collapsible) — for power users */}
            {/* ================================================= */}

            <details className="mt-8">
              <summary className="
              cursor-pointer
              select-none
              text-sm
              font-semibold
              text-gray-300
              bg-zinc-900
              border
              border-zinc-700
              rounded-xl
              px-4
              py-3
              hover:bg-zinc-800
              ">
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
              bg-zinc-900
              p-4
              rounded-2xl
              border
              border-zinc-700
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
              bg-zinc-900
              p-4
              rounded-2xl
              border
              border-zinc-700
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
            bg-zinc-900
            p-5
            rounded-2xl
            border
            border-zinc-700
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
                    <div className="border border-zinc-800 rounded-2xl p-5 bg-zinc-900 space-y-5">

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
                                <span className="px-3 py-1.5 rounded-lg bg-zinc-800 text-white font-mono text-sm font-semibold">
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
                          <div className="bg-zinc-800 rounded-xl p-3 text-center">
                            <p className="text-xs text-zinc-500 mb-1">Total Growth</p>
                            <p className="text-2xl font-bold text-cyan-400">
                              {v.total_growth}
                            </p>
                          </div>
                        )}

                        {rbiMatch !== undefined && (
                          <div className="bg-zinc-800 rounded-xl p-3 text-center">
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
      bg-zinc-900
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
    bg-zinc-900
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
    <div className="bg-zinc-900 p-4 rounded-2xl border border-zinc-700">
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
    <div className="mt-8 bg-zinc-900 p-5 rounded-2xl border border-zinc-700">

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
          <div className="bg-zinc-900 p-4 rounded-2xl border border-zinc-700">
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

function ExplainPanel({ result }: { result: PredictResponse }) {

  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [source, setSource] = useState<"llm" | "template" | null>(null);
  const [error, setError] = useState<string>("");

  const handleExplain = async () => {
    try {
      setLoading(true);
      setError("");

      const response = await axios.post<ExplainResponse>(
        "http://127.0.0.1:8000/explain",
        result,
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

  return (
    <div className="mt-8 bg-zinc-900 p-5 rounded-2xl border border-zinc-700">

      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">Explain with AI</h3>
        {source && (
          <span
            className={`text-xs font-bold px-3 py-1 rounded-full border ${
              source === "llm"
                ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/40"
                : "bg-zinc-700/40 text-gray-300 border-zinc-600"
            }`}
          >
            {source === "llm" ? "AI GENERATED" : "RULE-BASED SUMMARY"}
          </span>
        )}
      </div>

      {!explanation && (
        <p className="text-sm text-gray-400 mb-4">
          Get a plain-language summary of why this verdict was reached, plus
          manual checks you can do by hand. Designed to be read aloud.
        </p>
      )}

      <button
        onClick={handleExplain}
        disabled={loading}
        className="
        bg-indigo-600
        hover:bg-indigo-500
        disabled:opacity-60
        transition-all
        duration-300
        px-5
        py-2.5
        rounded-xl
        text-sm
        font-semibold
        "
      >
        {loading
          ? "Generating explanation..."
          : explanation
            ? "Regenerate explanation"
            : "Explain this result"}
      </button>

      {error && (
        <p className="text-sm text-red-400 mt-3">{error}</p>
      )}

      {explanation && (
        <div className="mt-5 space-y-5">

          {/* ---- Summary ---- */}
          <div>
            <p className="text-xs text-gray-500 mb-1 uppercase tracking-wider">
              Summary
            </p>
            <p className="text-base text-gray-200 leading-relaxed">
              {explanation.summary}
            </p>
          </div>

          {/* ---- Reasons ---- */}
          {explanation.reasons.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider">
                Why
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
                Check it by hand
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
    bg-zinc-900
    border
    border-zinc-800
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
          border-zinc-700
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
          border-zinc-700
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

function VerdictBanner({ result }: { result: PredictResponse }) {

  const verdict = result.prediction ?? "SUSPICIOUS";
  const display = VERDICT_DISPLAY[verdict] ?? VERDICT_DISPLAY.SUSPICIOUS;
  const level = result.verification_level;

  const denomRaw = result.forensic_analysis?.denomination_classification?.value;
  const denom = typeof denomRaw === "string" ? denomRaw : null;

  const isUnverified = verdict === "UNVERIFIED";
  const isLimited = !isUnverified && level === "partial";

  return (
    <div className={`rounded-2xl border ${display.border} ${display.bg} p-6`}>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-gray-400 mb-1">
            Result
          </p>
          <h2 className={`text-4xl font-bold ${display.accent}`}>
            {display.headline}
            {denom && !isUnverified && (
              <span className="text-gray-200 text-2xl font-semibold ml-2">
                ₹{denom}
              </span>
            )}
          </h2>
        </div>

        {!isUnverified && (
          <div className="text-right">
            <p className="text-xs text-gray-400">Confidence</p>
            <p className={`text-2xl font-bold ${display.accent}`}>
              {result.confidence}
            </p>
          </div>
        )}
      </div>

      <p className="text-sm text-gray-300 mt-3 leading-relaxed">
        {display.sub}
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
            How to get a good photo:
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
    <div className="mt-8 bg-zinc-900 border border-zinc-700 rounded-2xl p-5">

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