/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // In local dev (no NEXT_PUBLIC_API_URL), proxy /api/* to the FastAPI
    // backend so the browser can use same-origin calls (CORS-free). In
    // production, set NEXT_PUBLIC_API_URL and the frontend calls the backend
    // directly — no proxy needed.
    const target = process.env.NEXT_PUBLIC_API_URL;
    if (target) return [];
    return [{ source: "/api/:path*", destination: "http://localhost:8000/api/:path*" }];
  },
};

export default nextConfig;
