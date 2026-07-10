/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        rail:    { bg: "#1A1A2E", active: "#7C3AED", hover: "#252547", text: "#94A3B8" },
        sidebar: { bg: "#F0EEFF", card: "#FFFFFF", border: "#E0D9FF", title: "#1A1A2E", sub: "#6B7280" },
        chat:    { bg: "#FAFAFA", user: "#EDE9FE", ai: "#FFFFFF", border: "#E8E0FF" },
        accent:  { DEFAULT: "#7C3AED", hover: "#6D28D9", light: "#EDE9FE", blue: "#3B82F6" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        card: "0 2px 16px rgba(124,58,237,0.08)",
        "card-hover": "0 4px 24px rgba(124,58,237,0.15)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

