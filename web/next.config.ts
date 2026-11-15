import type { NextConfig } from "next";

// Static export: `yarn build` writes plain HTML/JS to out/, so it can be hosted on GitHub Pages or any static host.
// For a project page (user.github.io/<repo>/) build with NEXT_PUBLIC_BASE_PATH=/<repo>.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
