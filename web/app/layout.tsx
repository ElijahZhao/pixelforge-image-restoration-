import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PixelForge — AI Image Restoration",
  description:
    "Super-resolution and low-light enhancement powered by self-trained PyTorch models. A computer-vision portfolio project.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
