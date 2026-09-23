import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Pi'nin yerel ağındaki diğer cihazlardan da erişilebilsin diye
    proxy: {
      // Geliştirme sırasında /api istekleri backend'e (uvicorn, 8000 portu) yönlendirilir.
      // main.py'deki CORS ayarı sayesinde bu proxy olmadan da çalışır,
      // ancak proxy ile aynı origin üzerinden gidildiği için daha temizdir.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
