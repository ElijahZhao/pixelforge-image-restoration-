/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // `ink` is used by globals.css / CompareSlider (bg-ink, text-ink).
        ink: "#0b0f1a",
      },
    },
  },
  plugins: [],
};
