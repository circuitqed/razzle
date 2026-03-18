import { defineConfig } from 'vite'
import { resolve } from 'path'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        'test-mcts': resolve(__dirname, 'test-mcts.html'),
      },
    },
  },
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
    allowedHosts: ['knightball.org'],
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
