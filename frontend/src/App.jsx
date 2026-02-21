import { useState, useRef, useCallback } from 'react'
import InfoPage from './InfoPage.jsx'

// ── Tool icon map ────────────────────────────────────────────────
const TOOL_ICONS = {
  get_weather: '🌤',
  set_alarm: '⏰',
  send_message: '📨',
  play_music: '🎵',
  set_timer: '⏱',
  create_reminder: '🔔',
  search_contacts: '🔍',
  open_app: '📱',
  make_call: '📞',
}

// ── Source badge ─────────────────────────────────────────────────
function SourceBadge({ source }) {
  const isLocal = source && source.includes('on-device')
  return (
    <span className={`badge ${isLocal ? 'badge--local' : 'badge--cloud'}`}>
      {isLocal ? '⚡ on-device' : '☁ cloud'}
    </span>
  )
}

// ── Action Card ──────────────────────────────────────────────────
function ActionCard({ result }) {
  if (!result) return null
  const { transcript, actions = [], function_calls = [], source, latency_ms = {} } = result

  return (
    <div className="action-card action-card--enter">
      <div className="action-card__header">
        <p className="action-card__transcript">"{transcript}"</p>
        <div className="action-card__meta">
          <SourceBadge source={source} />
          <span className="action-card__latency">{Math.round(latency_ms.total ?? 0)}ms</span>
        </div>
      </div>

      {actions.length > 0 ? (
        <ul className="action-card__actions">
          {actions.map((a, i) => (
            <li key={i} className={`action-item ${a.success ? '' : 'action-item--error'}`}>
              <span className="action-item__icon">
                {TOOL_ICONS[a.tool] ?? '✓'}
              </span>
              <span className="action-item__summary">{a.summary}</span>
            </li>
          ))}
        </ul>
      ) : function_calls.length > 0 ? (
        <ul className="action-card__actions">
          {function_calls.map((fc, i) => (
            <li key={i} className="action-item">
              <span className="action-item__icon">{TOOL_ICONS[fc.name] ?? '✓'}</span>
              <span className="action-item__summary">
                {fc.name}({Object.entries(fc.arguments).map(([k, v]) => `${k}: ${v}`).join(', ')})
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="action-card__empty">No actions taken</p>
      )}

      {latency_ms.transcription !== undefined && latency_ms.transcription > 0 && (
        <div className="action-card__latency-detail">
          <span>transcribe {Math.round(latency_ms.transcription)}ms</span>
          <span>route {Math.round(latency_ms.routing)}ms</span>
        </div>
      )}
    </div>
  )
}

// ── History Feed ─────────────────────────────────────────────────
function HistoryFeed({ history }) {
  if (history.length === 0) return null
  return (
    <div className="history">
      <p className="history__label">HISTORY</p>
      <ul className="history__list">
        {[...history].reverse().map((item, i) => {
          const isLocal = item.source && item.source.includes('on-device')
          return (
            <li key={i} className="history__item">
              <span className={`history__dot ${isLocal ? 'history__dot--local' : 'history__dot--cloud'}`} />
              <span className="history__text">"{item.transcript}"</span>
              <span className="history__time">{Math.round(item.latency_ms?.total ?? 0)}ms</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

// ── Mic Button ───────────────────────────────────────────────────
function MicButton({ onResult, onError, isLoading, setIsLoading }) {
  const [recording, setRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data)
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const formData = new FormData()
        formData.append('audio', blob, 'audio.webm')

        setIsLoading(true)
        try {
          const res = await fetch('/api/transcribe-and-act', { method: 'POST', body: formData })
          const data = await res.json()
          if (data.error) onError(data.error)
          else onResult(data)
        } catch (err) {
          onError('Failed to reach server')
        } finally {
          setIsLoading(false)
        }
      }

      mediaRecorder.start()
      setRecording(true)
    } catch (err) {
      onError('Microphone access denied')
    }
  }, [onResult, onError, setIsLoading])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
    }
  }, [recording])

  const handleClick = () => {
    if (isLoading) return
    if (recording) stopRecording()
    else startRecording()
  }

  return (
    <button
      className={`mic-btn ${recording ? 'mic-btn--recording' : ''} ${isLoading ? 'mic-btn--loading' : ''}`}
      onClick={handleClick}
      disabled={isLoading}
      aria-label={recording ? 'Stop recording' : 'Start recording'}
    >
      {isLoading ? (
        <span className="mic-btn__spinner" />
      ) : recording ? (
        <svg viewBox="0 0 24 24" fill="currentColor" width="36" height="36">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="currentColor" width="36" height="36">
          <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 2a2 2 0 0 0-2 2v6a2 2 0 0 0 4 0V5a2 2 0 0 0-2-2zm7 8a1 1 0 0 1 1 1 8 8 0 0 1-7 7.938V21h2a1 1 0 1 1 0 2H9a1 1 0 1 1 0-2h2v-1.062A8 8 0 0 1 4 12a1 1 0 1 1 2 0 6 6 0 0 0 12 0 1 1 0 0 1 1-1z" />
        </svg>
      )}
      <span className="mic-btn__label">
        {isLoading ? 'Processing...' : recording ? 'Tap to stop' : 'Tap to speak'}
      </span>
    </button>
  )
}

// ── Text Input ───────────────────────────────────────────────────
function TextInput({ onResult, onError, isLoading, setIsLoading }) {
  const [text, setText] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim() || isLoading) return
    setIsLoading(true)
    try {
      const res = await fetch('/api/text-command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      const data = await res.json()
      if (data.error) onError(data.error)
      else onResult(data)
      setText('')
    } catch (err) {
      onError('Failed to reach server')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <form className="text-input" onSubmit={handleSubmit}>
      <input
        className="text-input__field"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Or type a command..."
        disabled={isLoading}
      />
      <button className="text-input__btn" type="submit" disabled={isLoading || !text.trim()}>
        <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
        </svg>
      </button>
    </form>
  )
}

// ── Main App ─────────────────────────────────────────────────────
export default function App() {
  const [latestResult, setLatestResult] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

  const handleResult = (data) => {
    setLatestResult(data)
    setHistory(prev => [...prev, data])
    setError(null)
  }

  const handleError = (msg) => {
    setError(msg)
    setTimeout(() => setError(null), 3000)
  }

  if (showInfo) {
    return (
      <div className="app">
        <div className="app__grid" aria-hidden="true" />
        <InfoPage onBack={() => setShowInfo(false)} />
      </div>
    )
  }

  return (
    <div className="app">
      {/* Background grid */}
      <div className="app__grid" aria-hidden="true" />

      {/* Header */}
      <header className="header">
        <div className="header__logo">
          <span className="header__logo-icon">◈</span>
          <span className="header__logo-text">DrivR</span>
        </div>
        <div className="header__row">
          <p className="header__tagline">Speak it. Done.</p>
          <button className="header__info-btn" onClick={() => setShowInfo(true)}>about</button>
        </div>
      </header>

      {/* Main content */}
      <main className="main">
        <MicButton
          onResult={handleResult}
          onError={handleError}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />

        {error && (
          <div className="error-toast">{error}</div>
        )}

        <ActionCard result={latestResult} />

        <TextInput
          onResult={handleResult}
          onError={handleError}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />

        <HistoryFeed history={history} />
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>FunctionGemma · Cactus · Gemini</span>
      </footer>
    </div>
  )
}