import { BOOKING_STATUS, CALL_STATUS, CHANNEL } from '../labels'
import { formatDay, formatPhone, formatTime, weekdayName } from '../time'
import type { Booking, CallSummary, DaySchedule } from '../types'
import { Notice } from './Notice'
import { StatusText } from './StatusText'

interface Props {
  schedule: DaySchedule
  isToday: boolean
  timezone: string
  selectedId: string | null
  onSelect: (id: string) => void
}

// The day as a front-desk register: one ruled line per entry, in time order.
export function DayList({ schedule, isToday, timezone, selectedId, onSelect }: Props) {
  const { date, open, bookings, otherCalls } = schedule
  // Newest first, so a call in progress is at the top.
  const calls = [...otherCalls].sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt))

  return (
    <>
      <h2 className="day-title">
        {formatDay(date)}
        {isToday && <span className="tag">Today</span>}
      </h2>

      {bookings.length > 0 ? (
        <section aria-labelledby="appointments-heading">
          <h3 id="appointments-heading" className="section-title">
            Appointments ({bookings.length})
          </h3>
          <ol className="register">
            {bookings.map((booking) => (
              <BookingRow
                key={booking.id}
                booking={booking}
                timezone={timezone}
                selected={booking.id === selectedId}
                onSelect={onSelect}
              />
            ))}
          </ol>
        </section>
      ) : open ? (
        <Notice title={`No appointments on ${formatDay(date)}`}>
          Nothing is booked yet. Bookings made by phone or WhatsApp appear here as they happen.
        </Notice>
      ) : (
        <Notice title={`Closed on ${weekdayName(date)}s`}>
          The agent does not offer appointments on closed days.
        </Notice>
      )}

      {calls.length > 0 && (
        <section aria-labelledby="calls-heading">
          <h3 id="calls-heading" className="section-title">
            Calls with no booking ({calls.length})
          </h3>
          <ol className="register">
            {calls.map((call) => (
              <CallRow
                key={call.id}
                call={call}
                timezone={timezone}
                selected={call.id === selectedId}
                onSelect={onSelect}
              />
            ))}
          </ol>
        </section>
      )}
    </>
  )
}

interface RowProps {
  timezone: string
  selected: boolean
  onSelect: (id: string) => void
}

function BookingRow({ booking, timezone, selected, onSelect }: RowProps & { booking: Booking }) {
  const status = BOOKING_STATUS[booking.status]
  return (
    <li>
      <button
        type="button"
        className={`entry${booking.status === 'cancelled' ? ' struck' : ''}`}
        data-entry={booking.id}
        aria-current={selected ? 'true' : undefined}
        onClick={() => onSelect(booking.id)}
      >
        <span className="entry-time">{formatTime(booking.slotStart, timezone)}</span>
        <span className="entry-main">
          <span className="entry-name">{booking.customerName}</span>
          <span className="entry-detail">
            {booking.serviceName}
            {booking.staffName ? ` with ${booking.staffName}` : ''}
          </span>
        </span>
        <StatusText {...status} />
      </button>
    </li>
  )
}

function CallRow({ call, timezone, selected, onSelect }: RowProps & { call: CallSummary }) {
  const status = CALL_STATUS[call.status]
  const detail = call.summary ?? (call.status === 'in_progress' ? 'Call in progress' : CHANNEL[call.channel])
  return (
    <li>
      <button
        type="button"
        className="entry"
        data-entry={call.id}
        aria-current={selected ? 'true' : undefined}
        onClick={() => onSelect(call.id)}
      >
        <span className="entry-time">{formatTime(call.startedAt, timezone)}</span>
        <span className="entry-main">
          <span className="entry-name">{call.fromPhone ? formatPhone(call.fromPhone) : 'Unknown number'}</span>
          <span className="entry-detail">{detail}</span>
        </span>
        <StatusText {...status} />
      </button>
    </li>
  )
}
