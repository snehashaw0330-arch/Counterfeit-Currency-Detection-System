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

type PredictResponse = {
  status: "success" | "error";
  prediction?: "REAL" | "FAKE" | "SUSPICIOUS";
  confidence?: string;
  raw_prediction?: number;
  model_verdict?: "REAL" | "FAKE";
  model_confidence?: string;
  forensic_score?: number;
  forensic_pass_count?: number;
  forensic_total_checks?: number;
  forensic_analysis?: ForensicAnalysis;
  message?: string;
};

const VERDICT_COLOR: Record<string, string> = {
  REAL: "text-green-400",
  FAKE: "text-red-400",
  SUSPICIOUS: "text-yellow-400",
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

            <h2 className="
            text-3xl
            font-bold
            mb-8
            text-center
            ">

              Detection Result

            </h2>

            {/* RESULT GRID */}

            <div className="
            grid
            grid-cols-1
            md:grid-cols-2
            gap-6
            ">

              {/* FINAL VERDICT */}

              <div className="
              bg-zinc-900
              p-5
              rounded-2xl
              border
              border-zinc-700
              ">

                <h3 className="text-gray-400 mb-2">
                  Final Verdict
                </h3>

                <p
                  className={`
                  text-3xl
                  font-bold
                  ${VERDICT_COLOR[result.prediction ?? ""]
                    ?? "text-gray-300"}
                  `}
                >
                  {result.prediction}
                </p>

                <p className="text-xs text-gray-500 mt-2">
                  Model + Forensic combined
                </p>

              </div>

              {/* CONFIDENCE */}

              <div className="
              bg-zinc-900
              p-5
              rounded-2xl
              border
              border-zinc-700
              ">

                <h3 className="text-gray-400 mb-2">
                  Confidence
                </h3>

                <p className="
                text-3xl
                font-bold
                text-yellow-400
                ">
                  {result.confidence}
                </p>

                <p className="text-xs text-gray-500 mt-2">
                  Forensic pass: {result.forensic_pass_count ?? 0}
                  /{result.forensic_total_checks ?? 0}
                </p>

              </div>

            </div>

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
          </div>
        )}
      </div>
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