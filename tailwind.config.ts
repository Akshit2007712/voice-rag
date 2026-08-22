import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Design token system — "Arcade Terminal" palette.
        // Sky is the world, cream is the paper/panel, ink is the line art.
        // Coin/pipe/brick/alert are functional accents borrowed from a
        // pixel-platformer's HUD language (score, go, warn, danger).
        sky: {
          DEFAULT: "#5C86E8",
          deep: "#2B4C9E",
          night: "#16214A",
        },
        cream: {
          DEFAULT: "#FFF7E0",
          panel: "#FFFDF6",
          dim: "#F3E7C4",
        },
        ink: {
          DEFAULT: "#1C1B2E",
          soft: "#3C3B57",
        },
        coin: {
          DEFAULT: "#FFC93C",
          deep: "#E89A1C",
        },
        pipe: {
          DEFAULT: "#1AA35C",
          deep: "#0E7A42",
        },
        brick: {
          DEFAULT: "#C2481B",
          deep: "#8F3210",
        },
        alert: {
          DEFAULT: "#E5453A",
          deep: "#AC2A22",
        },
      },
      fontFamily: {
        pixel: ["var(--font-pixel)", "monospace"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        // Hard-edged, offset "step" shadows instead of blurred ones —
        // reads as a pixel-art drop shadow, not a modern soft shadow.
        pixel: "4px 4px 0 0 #1C1B2E",
        "pixel-sm": "2px 2px 0 0 #1C1B2E",
        "pixel-lg": "6px 6px 0 0 #1C1B2E",
        "pixel-coin": "3px 3px 0 0 #8F5A00",
        "pixel-inset": "inset 3px 3px 0 0 rgba(0,0,0,0.18)",
      },
      borderRadius: {
        none: "0px",
        px2: "2px",
      },
      keyframes: {
        blink: {
          "0%, 49%": { opacity: "1" },
          "50%, 100%": { opacity: "0" },
        },
        bob: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-6px)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.9)", opacity: "0.9" },
          "100%": { transform: "scale(1.9)", opacity: "0" },
        },
        "bar-1": {
          "0%, 100%": { height: "20%" },
          "50%": { height: "90%" },
        },
        "bar-2": {
          "0%, 100%": { height: "60%" },
          "50%": { height: "15%" },
        },
        "bar-3": {
          "0%, 100%": { height: "35%" },
          "50%": { height: "100%" },
        },
        "bar-4": {
          "0%, 100%": { height: "80%" },
          "50%": { height: "30%" },
        },
        "coin-flip": {
          "0%": { transform: "scaleX(1)" },
          "50%": { transform: "scaleX(0.1)" },
          "100%": { transform: "scaleX(1)" },
        },
        marquee: {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "32px 0" },
        },
      },
      animation: {
        blink: "blink 1s step-start infinite",
        bob: "bob 2.2s ease-in-out infinite",
        "pulse-ring": "pulse-ring 1.4s cubic-bezier(0.2,0.6,0.4,1) infinite",
        "bar-1": "bar-1 0.55s ease-in-out infinite",
        "bar-2": "bar-2 0.5s ease-in-out infinite",
        "bar-3": "bar-3 0.65s ease-in-out infinite",
        "bar-4": "bar-4 0.45s ease-in-out infinite",
        "coin-flip": "coin-flip 0.6s linear infinite",
        marquee: "marquee 1.2s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
