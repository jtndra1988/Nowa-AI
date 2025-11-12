/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand core
        nowa: {
          primary: "#2563EB", // vivid blue
          secondary: "#22C55E", // signal green
          accent: "#F97316",   // orange accent
          pink: "#EC4899",
          // light surfaces
          surface: "#F5F7FB",
          // dark surfaces (already look good in your dark theme)
          "surface-dark": "#020817",
        },
        // Neutral scales tuned for light & dark
        slate: {
          25: "#F9FAFB",
          50: "#F8FAFC",
          75: "#EEF2FF",
          850: "#020817",
        },
      },
      boxShadow: {
        "soft-lg": "0 14px 45px rgba(15,23,42,0.06)",
        "soft-sm": "0 6px 18px rgba(15,23,42,0.04)",
      },
      backgroundImage: {
        "nowa-light":
          "radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent), radial-gradient(circle at top right, rgba(236,72,153,0.06), transparent)",
        "nowa-dark":
          "radial-gradient(circle at top, rgba(56,189,248,0.18), transparent), radial-gradient(circle at right, rgba(129,140,248,0.14), transparent)",
      },
    },
  },
  plugins: [],
};
