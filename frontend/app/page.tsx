"use client";

import { useState } from "react";
import axios from "axios";

type ForensicCheck = {
  status: "PASS" | "FAIL" | "INFO";
  details: string;
  value?: string | null;
};

type ForensicAnalysis = {
  uv_light_detection: ForensicCheck;
  watermark_detection: ForensicCheck;
  ocr_serial_number: ForensicCheck;
  gandhi_face_analysis: ForensicCheck;
  security_thread_detection: ForensicCheck;
  hologram_detection: ForensicCheck;
  denomination_classification: ForensicCheck;
  modular_ai_pipeline: ForensicCheck;
};

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
                  result?.forensic_analysis
                    ?.ocr_serial_number?.value
                  || "Not Detected"
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

                <FeatureCard
                  title="Hologram Detection"
                  check={result?.forensic_analysis?.hologram_detection}
                />

                <FeatureCard
                  title="Denomination Classification"
                  check={result?.forensic_analysis?.denomination_classification}
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

      {check?.value && (
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