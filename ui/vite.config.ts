import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { analyzePlugin } from "./analyzePlugin";

export default defineConfig({
    plugins: [react(), tailwindcss(), analyzePlugin()],
});
