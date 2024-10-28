/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Move this outside the theme object
  theme: {
    extend: {},
  },
  plugins: [],
}
