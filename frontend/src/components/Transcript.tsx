import { TOOL_NOTE } from '../labels'
import { formatClock } from '../time'
import type { Call, Turn } from '../types'

interface Props {
  call: Call
  turns: Turn[] // the lines to show: for a live call, only those that have arrived
  live: boolean
}

export function Transcript({ call, turns, live }: Props) {
  const started = Date.parse(call.startedAt)

  return (
    <section aria-labelledby="transcript-heading">
      <h3 id="transcript-heading" className="section-title">
        Transcript
      </h3>
      {/* A call in progress adds lines as they are said; polite, so it never talks over the screen reader. */}
      <ol className="transcript" aria-live={live ? 'polite' : undefined}>
        {turns.map((turn) => {
          // The model answered, but the voice never reached the caller.
          const unheard =
            turn.role === 'agent' && turn.timings !== null && turn.timings.sttFinalMs !== null && turn.timings.totalMs === null
          return (
            <li key={turn.index} className={`turn turn-${turn.role}`}>
              <p className="turn-who">
                <span>{turn.role === 'agent' ? 'Agent' : 'Caller'}</span>
                <span className="turn-time">{formatClock(Date.parse(turn.at) - started)}</span>
              </p>
              {turn.tools.map((tool) => (
                <p key={tool} className="turn-note">
                  {TOOL_NOTE[tool]}
                </p>
              ))}
              <p className="turn-text">{turn.text}</p>
              {turn.bargeIn && <p className="turn-note">The caller spoke over this reply.</p>}
              {unheard && <p className="turn-note turn-note-failed">The caller never heard this reply.</p>}
            </li>
          )
        })}
        {live && <li className="turn-listening">Listening</li>}
      </ol>
    </section>
  )
}
