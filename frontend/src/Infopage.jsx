export default function InfoPage({ onBack }) {
  return (
    <div className="info-page">
      <button className="info-back-btn" onClick={onBack}>
        ← Back
      </button>

      <div className="info-hero">
        <span className="info-hero__icon">◈</span>
        <h1 className="info-hero__title">DrivR</h1>
        <p className="info-hero__sub">Your hands-free driver assistant</p>
      </div>

      <section className="info-section">
        <p className="info-section__body">
          DrivR lets you control your phone entirely by voice while driving —
          no tapping, no glancing at the screen. Speak a command, and it's done.
          Set alarms, send messages, check the weather, play music — all without
          taking your eyes off the road.
        </p>
      </section>

      <div className="info-divider" />

      <section className="info-section">
        <h2 className="info-section__title">How it works</h2>
        <div className="info-steps">
          <div className="info-step">
            <span className="info-step__num">01</span>
            <div>
              <p className="info-step__label">You speak</p>
              <p className="info-step__desc">Your voice is captured on-device and transcribed instantly using Whisper — no audio ever leaves your phone.</p>
            </div>
          </div>
          <div className="info-step">
            <span className="info-step__num">02</span>
            <div>
              <p className="info-step__label">It routes intelligently</p>
              <p className="info-step__desc">Simple commands are handled locally in milliseconds. Complex ones escalate to the cloud only when needed.</p>
            </div>
          </div>
          <div className="info-step">
            <span className="info-step__num">03</span>
            <div>
              <p className="info-step__label">Action is taken</p>
              <p className="info-step__desc">The right tool is called — alarm, message, weather, music — and you hear the confirmation.</p>
            </div>
          </div>
        </div>
      </section>

      <div className="info-divider" />

      <section className="info-section">
        <h2 className="info-section__title">The technology</h2>

        <div className="info-tech-cards">
          <div className="info-tech-card">
            <div className="info-tech-card__header">
              <span className="info-tech-card__badge info-tech-card__badge--local">on-device</span>
              <span className="info-tech-card__name">Cactus Engine</span>
            </div>
            <p className="info-tech-card__desc">
              A lightweight AI inference engine that runs large language models
              directly on your device — no internet required. Optimised for ARM
              processors in modern smartphones and Macs, delivering sub-50ms
              response times with minimal battery drain.
            </p>
          </div>

          <div className="info-tech-card">
            <div className="info-tech-card__header">
              <span className="info-tech-card__badge info-tech-card__badge--local">on-device</span>
              <span className="info-tech-card__name">FunctionGemma</span>
            </div>
            <p className="info-tech-card__desc">
              A compact 270M-parameter model by Google DeepMind, purpose-built
              for tool calling and agentic tasks. Runs entirely on-device via
              Cactus, making it ideal for fast, private, offline-capable
              command routing.
            </p>
          </div>

          <div className="info-tech-card">
            <div className="info-tech-card__header">
              <span className="info-tech-card__badge info-tech-card__badge--local">on-device</span>
              <span className="info-tech-card__name">Whisper</span>
            </div>
            <p className="info-tech-card__desc">
              OpenAI's speech recognition model, running locally via Cactus.
              Transcribes your voice in real time with no round-trip to the
              cloud — keeping your conversations private and working even
              without a data connection.
            </p>
          </div>

          <div className="info-tech-card">
            <div className="info-tech-card__header">
              <span className="info-tech-card__badge info-tech-card__badge--cloud">cloud</span>
              <span className="info-tech-card__name">Gemini Flash</span>
            </div>
            <p className="info-tech-card__desc">
              Google DeepMind's frontier model, used as a fallback for complex
              or ambiguous commands that benefit from deeper reasoning. Only
              invoked when the on-device model isn't confident — minimising
              latency and data usage.
            </p>
          </div>
        </div>
      </section>

      <div className="info-divider" />

      <section className="info-section">
        <h2 className="info-section__title">Why local-first?</h2>
        <div className="info-stats">
          <div className="info-stat">
            <span className="info-stat__value">&lt;50ms</span>
            <span className="info-stat__label">on-device latency</span>
          </div>
          <div className="info-stat">
            <span className="info-stat__value">~80%</span>
            <span className="info-stat__label">commands handled locally</span>
          </div>
          <div className="info-stat">
            <span className="info-stat__value">0 bytes</span>
            <span className="info-stat__label">audio sent to cloud</span>
          </div>
        </div>
        <p className="info-section__body" style={{marginTop: '16px'}}>
          Running AI locally means faster responses, better privacy, and
          continued functionality even in tunnels or areas with poor signal —
          exactly what you need behind the wheel.
        </p>
      </section>

      <div className="info-footer">
        Built at the Cactus × Google DeepMind Hackathon · Singapore 2026
      </div>
    </div>
  )
}