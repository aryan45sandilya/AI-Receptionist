import { useEffect, useRef, useState } from 'react'
import { SOURCE } from '../labels'
import { dateIn, formatDay, formatPhone, formatTime } from '../time'
import type { Booking } from '../types'

interface Props {
  booking: Booking
  timezone: string
  onCancel: (bookingId: string) => void
}

export function BookingBlock({ booking, timezone, onCancel }: Props) {
  const [confirming, setConfirming] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const keepButton = useRef<HTMLButtonElement>(null)
  const feedback = useRef<HTMLParagraphElement>(null)

  // Land on the safe choice, and when the question goes away, on its answer, so
  // keyboard focus is never left on a button that no longer exists.
  useEffect(() => {
    if (confirming) keepButton.current?.focus()
  }, [confirming])
  useEffect(() => {
    if (cancelled) feedback.current?.focus()
  }, [cancelled])

  const day = formatDay(dateIn(timezone, booking.slotStart))
  const start = formatTime(booking.slotStart, timezone)
  const end = formatTime(booking.slotEnd, timezone)

  return (
    <section aria-labelledby="booking-heading">
      <h3 id="booking-heading" className="section-title">
        Appointment
      </h3>
      <dl className="facts">
        <dt>Service</dt>
        <dd>{booking.serviceName}</dd>
        {booking.staffName && (
          <>
            <dt>With</dt>
            <dd>{booking.staffName}</dd>
          </>
        )}
        <dt>When</dt>
        <dd>
          {day}, {start} to {end}
        </dd>
        <dt>Phone</dt>
        <dd>{formatPhone(booking.customerPhone)}</dd>
        <dt>Reference</dt>
        <dd>{booking.reference}</dd>
        <dt>Booked by</dt>
        <dd>{SOURCE[booking.source]}</dd>
      </dl>

      <div className="actions">
        <a className="btn" href={`tel:${booking.customerPhone}`}>
          Call back
        </a>
        {booking.status === 'booked' && !confirming && (
          <button type="button" className="btn danger" onClick={() => setConfirming(true)}>
            Cancel booking
          </button>
        )}
      </div>

      {confirming && (
        <div className="confirm" role="group" aria-labelledby="confirm-question">
          <p id="confirm-question" className="confirm-question">
            Cancel {booking.customerName}&rsquo;s {booking.serviceName.toLowerCase()} at {start}?
          </p>
          <div className="actions">
            <button
              type="button"
              className="btn danger filled"
              onClick={() => {
                onCancel(booking.id)
                setConfirming(false)
                setCancelled(true)
              }}
            >
              Yes, cancel booking
            </button>
            <button type="button" className="btn secondary" ref={keepButton} onClick={() => setConfirming(false)}>
              Keep booking
            </button>
          </div>
        </div>
      )}

      <p className="feedback" role="status" tabIndex={-1} ref={feedback}>
        {cancelled ? 'Booking cancelled.' : ''}
      </p>
    </section>
  )
}
