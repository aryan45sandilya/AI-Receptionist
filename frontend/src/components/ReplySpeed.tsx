import { isVoice } from '../labels'
import { formatSeconds } from '../time'
import type { Call, ReplyTimings, Turn } from '../types'

interface Props {
  call: Call
  turns: Turn[]
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

function seconds(ms: number | null): string {
  return ms === null ? 'none' : formatSeconds(ms)
}

// A time the caller was waiting for that never came, which is a failure, not a gap.
function Heard({ ms }: { ms: number | null }) {
  return <td className={ms === null ? 'none' : undefined}>{ms === null ? 'Not heard' : formatSeconds(ms)}</td>
}

// How long the caller waited, reply by reply. These are the same numbers the
// pipeline writes to Postgres for every turn.
export function ReplySpeed({ call, turns }: Props) {
  const voice = isVoice(call.channel)

  // A voice reply has a transcript step; a text reply has only its total.
  const replies: { index: number; timings: ReplyTimings }[] = []
  for (const turn of turns) {
    const t = turn.timings
    if (t && (voice ? t.sttFinalMs !== null : t.totalMs !== null)) replies.push({ index: turn.index, timings: t })
  }
  if (replies.length === 0) return null

  const heard = replies.map((r) => r.timings.totalMs).filter((ms): ms is number => ms !== null)
  const headline =
    heard.length > 0
      ? `Reply speed: median ${formatSeconds(median(heard))}, slowest ${formatSeconds(Math.max(...heard))}`
      : 'Reply speed: no reply was heard'

  return (
    <details className="speed">
      <summary>{headline}</summary>
      <p className="speed-note">
        {voice
          ? 'Seconds after the caller stopped speaking. When the agent looks something up it says a few words first, so the first sound comes before the answer.'
          : 'Seconds from the message arriving to the reply being sent. Text and voice replies are not comparable.'}
      </p>
      <div className="table-wrap" role="region" aria-label="Reply speed for each reply" tabIndex={0}>
        <table>
          <thead>
            <tr>
              <th scope="col">Reply</th>
              {voice ? (
                <>
                  <th scope="col">Transcript ready</th>
                  <th scope="col">First word written</th>
                  <th scope="col">First sound</th>
                  <th scope="col">Answer starts</th>
                </>
              ) : (
                <th scope="col">Reply sent</th>
              )}
            </tr>
          </thead>
          <tbody>
            {replies.map(({ index, timings: t }, position) => (
              <tr key={index}>
                <th scope="row">{position + 1}</th>
                {voice ? (
                  <>
                    <td>{seconds(t.sttFinalMs)}</td>
                    <td>{seconds(t.llmFirstTokenMs)}</td>
                    <Heard ms={t.fillerMs ?? t.totalMs} />
                    <Heard ms={t.totalMs} />
                  </>
                ) : (
                  <td>{seconds(t.totalMs)}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
