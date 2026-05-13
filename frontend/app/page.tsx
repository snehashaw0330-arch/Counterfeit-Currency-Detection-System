"use client";

import { useState } from "react";
import axios from "axios";

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
    useState<any>(null);

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

              {/* PREDICTION */}

              <div className="
              bg-zinc-900
              p-5
              rounded-2xl
              border
              border-zinc-700
              ">

                <h3 className="
                text-gray-400
                mb-2
                ">

                  Prediction

                </h3>

                <p
                  className={`
                  text-3xl
                  font-bold
                  ${
                    result.prediction ===
                    "REAL"
                    ? "text-green-400"
                    : "text-red-400"
                  }
                  `}
                >

                  {result.prediction}

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

                <h3 className="
                text-gray-400
                mb-2
                ">

                  Confidence

                </h3>

                <p className="
                text-3xl
                font-bold
                text-yellow-400
                ">

                  {result.confidence}

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
              ">

                {
                  result.serial_number
                  || "Not Available"
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
                />

                <FeatureCard
                  title="Watermark Detection"
                />

                <FeatureCard
                  title="OCR Serial Number"
                />

                <FeatureCard
                  title="Gandhi Face Analysis"
                />

                <FeatureCard
                  title="Security Thread Detection"
                />

                <FeatureCard
                  title="Hologram Detection"
                />

                <FeatureCard
                  title="Denomination Classification"
                />

                <FeatureCard
                  title="Modular AI Pipeline"
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

function FeatureCard({
  title
}: {
  title: string
}) {

  return (

    <div className="
    bg-zinc-900
    p-4
    rounded-xl
    border
    border-zinc-700
    hover:border-green-500
    transition-all
    duration-300
    ">

      {title}

    </div>
  );
}