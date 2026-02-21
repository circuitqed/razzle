import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          // Copy all WASM variants so ONNX Runtime can pick the best one for the device
          src: 'node_modules/onnxruntime-web/dist/ort-wasm*',
          dest: '.',
        },
      ],
    }),
  ],
  worker: {
    format: 'es',
  },
  server: {
    port: 7492,
    host: true,
    allowedHosts: ['razzledazzle.lazybrains.com'],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        rewrite: (path) => path.replace(/^\/ws/, ''),
      },
    },
  },
})
