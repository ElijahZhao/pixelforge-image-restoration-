"use client";

import { useRef, useState } from "react";
import CompareSlider from "@/components/CompareSlider";
import { predict, type PredictResult } from "@/lib/api";

type Task = "sr" | "lowlight";

export default function Home() {
  const [task, setTask] = useState<Task>("sr");
  const [scale, setScale] = useState(2);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = (f: File | null) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await predict(file, task, scale);
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-5xl px-5 py-10">
      <header className="mb-8">
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
          PixelForge
        </h1>
        <p className="mt-2 text-slate-300">
          Self-trained computer-vision models for{" "}
          <span className="text-sky-300">super-resolution</span> and{" "}
          <span className="text-fuchsia-300">low-light enhancement</span>.
          Upload an image and see the model in action.
        </p>
        <nav className="mt-3 text-sm">
          <a href="/method" className="text-sky-300 hover:underline">
            How it works →
          </a>
        </nav>
      </header>

      {/* Controls */}
      <section className="glass p-5 mb-6">
        <div className="flex flex-wrap gap-3 items-center">
          <div className="inline-flex rounded-lg border border-white/10 overflow-hidden">
            {(["sr", "lowlight"] as Task[]).map((t) => (
              <button
                key={t}
                onClick={() => setTask(t)}
                className={`px-4 py-2 text-sm ${
                  task === t ? "bg-sky-500/20 text-white" : "text-slate-300"
                }`}
              >
                {t === "sr" ? "Super-Resolution" : "Low-Light"}
              </button>
            ))}
          </div>

          {task === "sr" && (
            <div className="inline-flex rounded-lg border border-white/10 overflow-hidden">
              {[2, 4].map((s) => (
                <button
                  key={s}
                  onClick={() => setScale(s)}
                  className={`px-3 py-2 text-sm ${
                    scale === s ? "bg-sky-500/20 text-white" : "text-slate-300"
                  }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          )}

          <button
            onClick={() => inputRef.current?.click()}
            className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-sm"
          >
            Choose image
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => onPick(e.target.files?.[0] ?? null)}
          />
          {file && (
            <button
              onClick={run}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-sm font-medium"
            >
              {loading ? "Processing…" : "Enhance"}
            </button>
          )}
        </div>
        {preview && (
          <p className="mt-3 text-xs text-slate-400">
            Selected: {file?.name} · {task === "sr" ? `${scale}× upscale` : "low-light"}
          </p>
        )}
        {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
      </section>

      {/* Results */}
      {result ? (
        <section className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <span
              className={`text-xs px-2 py-1 rounded-full ${
                result.engine === "ml"
                  ? "bg-emerald-500/20 text-emerald-300"
                  : "bg-amber-500/20 text-amber-300"
              }`}
            >
              {result.engine === "ml" ? "Trained PyTorch model" : "Classical baseline"}
            </span>
            <span className="text-xs text-slate-400">{result.note}</span>
          </div>
          <CompareSlider
            before={result.before}
            after={result.after}
            beforeLabel={task === "sr" ? "Original (upscaled)" : "Low-light"}
            afterLabel={task === "sr" ? `Enhanced ${scale}×` : "Enhanced"}
          />
        </section>
      ) : preview ? (
        <section className="glass p-5 text-center text-slate-400 text-sm">
          Preview loaded — press <span className="text-white">Enhance</span> to run
          the model.
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="preview"
            className="mx-auto mt-3 max-h-72 rounded-lg border border-white/10"
          />
        </section>
      ) : (
        <section
          onClick={() => inputRef.current?.click()}
          className="glass p-12 text-center cursor-pointer hover:bg-white/10 transition"
        >
          <p className="text-slate-300">Click to upload an image</p>
          <p className="text-xs text-slate-500 mt-1">
            Try a small/blurry photo for super-resolution, or a dark photo for
            low-light enhancement.
          </p>
        </section>
      )}

      <footer className="mt-12 text-xs text-slate-500">
        Built as a graduate-admission CV portfolio. Models: SRCNN / SRResNet
        (super-resolution) and a U-Net (low-light), implemented in PyTorch. See{" "}
        <a href="/method" className="text-sky-300 hover:underline">
          method
        </a>
        .
      </footer>
    </main>
  );
}
