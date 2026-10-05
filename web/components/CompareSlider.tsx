"use client";

import { useRef, useState } from "react";

type Props = {
  before: string; // base64/data URL
  after: string; // base64/data URL
  beforeLabel?: string;
  afterLabel?: string;
};

// Interactive before/after image comparison slider.
//
// Alignment: the "after" layer is `block w-full` and therefore sizes the
// container via its intrinsic aspect ratio. The "before" overlay must occupy
// exactly the same box, or dragging the handle would compare pixels from
// different places. `object-cover` does NOT guarantee that: when the two images
// have different aspect ratios it CROPS the overlay, so an aligned comparison
// silently becomes an apples-to-oranges one. The backend already forces before
// and after to the same size, so we mirror that here with `object-fill` (and
// explicit inset-0) to make the overlay's box identical to the base layer's.
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

  const stopDrag = () => {
    dragging.current = false;
  };

  return (
    <div
      ref={ref}
      className="relative select-none overflow-hidden rounded-xl border border-white/10 touch-none"
      onPointerDown={(e) => {
        dragging.current = true;
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        updateFromClientX(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && updateFromClientX(e.clientX)}
      onPointerUp={stopDrag}
      // Without these, releasing the pointer outside the element (or the browser
      // cancelling the gesture, e.g. a touch scroll takes over) leaves
      // `dragging` stuck true and the handle keeps following the cursor.
      onPointerCancel={stopDrag}
      onPointerLeave={stopDrag}
      onLostPointerCapture={stopDrag}
    >
      {/* After (base layer): sizes the container via its aspect ratio */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`data:image/png;base64,${after}`} alt="After" className="block w-full" />

      {/* Before (clipped overlay): MUST match the base layer's box exactly */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${before}`}
          alt="Before"
          className="absolute inset-0 h-full w-full object-fill"
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
