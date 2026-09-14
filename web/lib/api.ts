// Client-side helper to call the FastAPI inference service.

export type PredictResult = {
  task: "sr" | "lowlight";
  scale: number;
  engine: "ml" | "classical";
  before: string; // base64 PNG
  after: string; // base64 PNG
  note: string;
};

export async function predict(
  file: File,
  task: "sr" | "lowlight",
  scale: number
): Promise<PredictResult> {
  // When NEXT_PUBLIC_API_URL is set (production), call the backend directly.
  // Otherwise use the same-origin /api path, which Next.js proxies to the
  // local FastAPI service (see next.config.mjs) — keeps the demo CORS-free.
  const base = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  const endpoint = base ? `${base}/api/predict` : `/api/predict`;
  const fd = new FormData();
  fd.append("image", file);
  fd.append("task", task);
  fd.append("scale", String(scale));
  const res = await fetch(endpoint, { method: "POST", body: fd });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Inference failed (${res.status}): ${msg}`);
  }
  return res.json();
}
