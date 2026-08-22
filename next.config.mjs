/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "https://voice-rag-backend-pdll.onrender.com/:path*",
      },
    ];
  },
};

export default nextConfig;
