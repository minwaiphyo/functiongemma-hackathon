# Frontend — DrivR

Vite + React single-page app for voice and text commands. Dark theme, mobile-first, minimal dependencies.

## Setup

```bash
cd frontend
npm install
```

## Run

```bash
npm run dev
```

Dev server runs at `http://localhost:5173`. API calls are proxied to the backend at `http://localhost:8000` via Vite config.

> **Important:** The backend must be running on port 8000 for API calls to work. See `backend/README.md` for setup.

## How It Works

1. **Mic Button** — Tap to record audio → sends WAV to `POST /api/transcribe-and-act`
2. **Text Input** — Type a command → sends to `POST /api/text-command`
3. **Action Card** — Displays the latest result with emoji, summary, source badge, and latency
4. **History Feed** — Scrollable list of past commands and results

## Files

| File | Purpose |
|---|---|
| `src/App.jsx` | Main app — all components (MicButton, TextInput, ActionCard, HistoryFeed) |
| `src/App.css` | Full styling — dark theme, animations, responsive layout |
| `src/main.jsx` | React entry point |
| `vite.config.js` | Vite config with `/api` proxy to backend |
| `index.html` | HTML shell with Google Fonts (Syne + Space Mono) |

## Proxy Config

All `/api/*` requests are forwarded to the backend:

```js
// vite.config.js
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```
