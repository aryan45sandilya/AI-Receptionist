import type { Async } from '../hooks'
import { useNow } from '../hooks'
import { addDays, formatClock, formatPhone, isDate } from '../time'
import type { Business, CallSummary } from '../types'

interface Props {
  business: Business
  date: string
  today: string
  live: Async<CallSummary | null>
  onDate: (date: string) => void
  onOpenLive: () => void
}

export function TopBar({ business, date, today, live, onDate, onOpenLive }: Props) {
  const call = live.status === 'ready' ? live.data : null
  const now = useNow(call ? 1000 : null)

  return (
    <header className="topbar">
      <h1 className="brand">{business.name}</h1>

      <div className="datebar" role="group" aria-label="Choose a day">
        <button type="button" className="btn secondary" onClick={() => onDate(addDays(date, -1))}>
          Previous day
        </button>
        <label className="datefield">
          <span className="sr-only">Day</span>
          <input
            type="date"
            value={date}
            onChange={(event) => {
              if (isDate(event.target.value)) onDate(event.target.value)
            }}
          />
        </label>
        <button type="button" className="btn secondary" onClick={() => onDate(addDays(date, 1))}>
          Next day
        </button>
        {date !== today && (
          <button type="button" className="btn" onClick={() => onDate(today)}>
            Go to today
          </button>
        )}
      </div>

      <div className="live-slot">
        {/* Announced when a call starts or ends, not every second the timer ticks. */}
        <p className="sr-only" role="status">
          {live.status === 'ready' ? (call ? 'A call is in progress.' : 'No call in progress.') : ''}
        </p>
        {call ? (
          <button type="button" className="live live-on" onClick={onOpenLive}>
            <span className="dot" aria-hidden="true" />
            <span>
              Live call {call.fromPhone ? formatPhone(call.fromPhone) : 'from an unknown number'}{' '}
              <span className="timer">{formatClock(now - Date.parse(call.startedAt))}</span>
            </span>
          </button>
        ) : (
          <p className="live live-off">
            {live.status === 'loading' && 'Checking for calls'}
            {live.status === 'error' && 'Could not check for a live call'}
            {live.status === 'ready' && 'No call in progress'}
          </p>
        )}
      </div>
    </header>
  )
}
