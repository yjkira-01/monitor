import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { output: { manualChunks(id) {
    if (id.includes('commonjsHelpers.js') || /\/node_modules\/(react|react-dom|react-is|scheduler)\//.test(id)) return 'framework'
    if (id.includes('/node_modules/three/')) return 'spatial'
    if (id.includes('/node_modules/recharts/')) return 'charts'
  } } } },
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
