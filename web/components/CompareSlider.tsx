"use client";

import { useRef, useState } from "react";

type Props = {
  before: string; // base64/data URL
  after: string; // base64/data URL
  beforeLabel?: string;
  afterLabel?: string;
};

// Interactive before/after image comparison slider.
// Uses clip-path so both layers stay perfectly aligned regardless of size.
export default function CompareSlider({
  before,
  after,
  beforeLabel = "Before",
  afterLabel = "After",
}: Props) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const p = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(0, Math.min(100, p)));
  };

  return (
    <div
      ref={ref}
      className="relative select-none overflow-hidden rounded-xl border border-white/10"
      onPointerDown={(e) => {
        dragging.current = true;
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        updateFromClientX(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && updateFromClientX(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
    >
      {/* After (base layer) */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`data:image/png;base64,${after}`} alt="After" className="block w-full" />

      {/* Before (clipped overlay) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${before}`}
          alt="Before"
          className="absolute inset-0 h-full w-full object-cover"
        />
      </div>

      {/* Labels */}
      <span className="absolute top-2 left-2 rounded-md bg-black/50 px-2 py-0.5 text-xs">
        {beforeLabel}
      </span>
      <span className="absolute top-2 right-2 rounded-md bg-black/50 px-2 py-0.5 text-xs">
        {afterLabel}
      </span>

      {/* Handle */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white/80"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-8 w-8 rounded-full bg-white text-ink grid place-items-center text-sm shadow">
          ↔
        </div>
      </div>
      <p className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[11px] text-white/70">
        drag to compare
      </p>
    </div>
  );
}
