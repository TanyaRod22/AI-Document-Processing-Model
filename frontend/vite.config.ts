import process from "node:process";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const netlify = process.env.NETLIFY === "true";
  const viteBase = process.env.VITE_API_BASE?.trim() ?? "";
  if (netlify && mode === "production" && !viteBase) {
    throw new Error(
      "Netlify production build requires VITE_API_BASE (your public API URL, e.g. https://….up.railway.app). " +
        "Add it under Site configuration → Environment variables, then redeploy.",
    );
  }

  return {
    plugins: [react()],
    server: {
      port: 5173,
    },
  };
});
