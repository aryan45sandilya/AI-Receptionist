import type { Ref } from 'react'
import { api } from '../api'
import { useAsync, useNow } from '../hooks'
import { BOOKING_STATUS, CALL_STATUS, CHANNEL } from '../labels'
import { formatDuration, formatMoment, formatPhone } from '../time'
import type { Booking, Call, CallSummary } from '../types'
import { BookingBlock } from './BookingBlock'
import { Notice } from './Notice'
import { ReplySpeed } from './ReplySpeed'
import { StatusText } from './StatusText'
import { Transcript } from './Transcript'

export type Selection = { kind: 'booking'; booking: Booking } | { kind: 'call'; call: CallSummary }

interface Props {
  selection: Selection
  timezone: string
  headingRef: Ref<HTMLHeadingElement>
  onBack: () => void
  onCancelBooking: (bookingId: string) => void
}

export function CallPanel({ selection, ...rest }: Props) {
  // A key per entry, so opening another one starts from a clean panel.
  const id = selection.kind === 'booking' ? selection.booking.id : selection.call.id
  return <SelectedPanel key={id} selection={selection} {...rest} />
}

function describe(call: CallSummary, timezone: string): string {
  const parts = [CHANNEL[call.channel], formatMoment(call.startedAt, timezone)]
  if (call.durationMs !== null) parts.push(formatDuration(call.durationMs))
  if (call.status === 'in_progress') parts.push('in progress')
  return parts.join(', ')
}

function SelectedPanel({ selection, timezone, headingRef, onBack, onCancelBooking }: Props) {
  const callId = selection.kind === 'booking' ? selection.booking.callId : selection.call.id
  const [loaded, retry] = useAsync(
    () => (callId ? api.loadCall(callId) : Promise.resolve(null)),
    callId ?? 'no call',
  )

  const title =
    selection.kind === 'booking'
      ? selection.booking.customerName
      : selection.call.fromPhone
        ? formatPhone(selection.call.fromPhone)
        : 'Unknown caller'
  const status =
    selection.kind === 'booking' ? BOOKING_STATUS[selection.booking.status] : CALL_STATUS[selection.call.status]

  return (
    <article className="panel" aria-labelledby="panel-heading">
      <button type="button" className="btn secondary back" onClick={onBack}>
        Back to the day list
      </button>

      <div className="panel-head">
        <h2 id="panel-heading" className="panel-title" ref={headingRef} tabIndex={-1}>
          {title}
        </h2>
        <StatusText {...status} />
      </div>

      {selection.kind === 'booking' && (
        <BookingBlock booking={selection.booking} timezone={timezone} onCancel={onCancelBooking} />
      )}
      {selection.kind === 'call' && <CallOutcome call={selection.call} timezone={timezone} />}

      {callId === null ? (
        <Notice title="No call to show">
          This appointment was entered at the front desk, so there is no conversation behind it.
        </Notice>
      ) : loaded.status === 'loading' ? (
        <p className="loading">Loading the call</p>
      ) : loaded.status === 'error' || loaded.data === null ? (
        <Notice
          tone="failed"
          title="Could not load this call"
          action={
            <button type="button" className="btn" onClick={retry}>
              Try again
            </button>
          }
        >
          Check that the server is running, then try again.
        </Notice>
      ) : (
        <Conversation call={loaded.data} timezone={timezone} showWhen={selection.kind === 'booking'} />
      )}
    </article>
  )
}

function Conversation({ call, timezone, showWhen }: { call: Call; timezone: string; showWhen: boolean }) {
  const live = call.status === 'in_progress'
  const now = useNow(live ? 1000 : null)
  // For a call in progress, only what has been said so far.
  const turns = live ? call.turns.filter((turn) => Date.parse(turn.at) <= now) : call.turns

  return (
    <>
      {showWhen && <p className="panel-meta">The call that made this booking: {describe(call, timezone)}</p>}
      <Transcript call={call} turns={turns} live={live} />
      <ReplySpeed call={call} turns={turns} />
    </>
  )
}

// What happened on a call that did not end in a booking, and what to do about it.
function CallOutcome({ call, timezone }: { call: CallSummary; timezone: string }) {
  return (
    <section aria-labelledby="outcome-heading">
      <h3 id="outcome-heading" className="section-title">
        The call
      </h3>
      <p className="panel-meta">{describe(call, timezone)}</p>
      {call.summary && <p className="summary">{call.summary}</p>}
      {call.handoffReason && (
        <p className="summary">
          <strong>Why it was handed to a person:</strong> {call.handoffReason}
        </p>
      )}
      {call.fromPhone && call.status !== 'in_progress' && (
        <div className="actions">
          <a className="btn" href={`tel:${call.fromPhone}`}>
            Call back
          </a>
        </div>
      )}
    </section>
  )
}
