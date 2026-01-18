/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        board: {
          light: '#f0d9b5',
          dark: '#b58863',
        },
        piece: {
          p1: '#3b82f6',
          p2: '#ef4444',
          ball: '#fbbf24',
        },
      },
    },
  },
  plugins: [],
}
