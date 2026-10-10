import { useEffect, useMemo, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { api } from './api'
import { CallPanel, type Selection } from './components/CallPanel'
import { DayList } from './components/DayList'
import { Notice } from './components/Notice'
import { TopBar } from './components/TopBar'
import { useAsync, useUrlState, useWide } from './hooks'
import { dateIn, formatDay, isDate } from './time'
import type { DaySchedule } from './types'

// What to show on a wide screen before anything is picked: a call in progress,
// else the next appointment still to come.
function defaultChoice(schedule: DaySchedule): string | null {
  const live = schedule.otherCalls.find((call) => call.status === 'in_progress')
  return (
    live?.id ??
    schedule.bookings.find((booking) => booking.status === 'booked')?.id ??
    schedule.bookings[0]?.id ??
    schedule.otherCalls[0]?.id ??
    null
  )
}

export default function App() {
  const { business } = api
  const [params, setParams] = useUrlState()
  const wide = useWide()
  const today = dateIn(business.timezone)
  const requested = params.get('date')
  const date = isDate(requested) ? requested : today
  const selectedId = params.get('item')

  const [day, retryDay] = useAsync(() => api.loadDay(date), date)
  const [live] = useAsync(() => api.loadLiveCall(), 'live')
  // Bookings cancelled in this session. With the API this becomes a request and a reload.
  const [cancelled, setCancelled] = useState<ReadonlySet<string>>(new Set())
  const detailHeading = useRef<HTMLHeadingElement>(null)

  const schedule = useMemo<DaySchedule | null>(() => {
    if (day.status !== 'ready') return null
    return {
      ...day.data,
      bookings: day.data.bookings.map((booking) =>
        cancelled.has(booking.id) ? { ...booking, status: 'cancelled' as const } : booking,
      ),
    }
  }, [day, cancelled])

  const selection = useMemo<Selection | null>(() => {
    if (!schedule || !selectedId) return null
    const booking = schedule.bookings.find((b) => b.id === selectedId)
    if (booking) return { kind: 'booking', booking }
    const call = schedule.otherCalls.find((c) => c.id === selectedId)
    return call ? { kind: 'call', call } : null
  }, [schedule, selectedId])

  const nothingOnThisDay = schedule !== null && schedule.bookings.length === 0 && schedule.otherCalls.length === 0

  useEffect(() => {
    document.title = `${business.name}, ${formatDay(date)}`
  }, [business.name, date])

  // Replace rather than push, so Back does not land on a screen that re-selects.
  useEffect(() => {
    if (!schedule || selectedId || !wide) return
    const choice = defaultChoice(schedule)
    if (choice) setParams({ item: choice }, 'replace')
  }, [schedule, selectedId, wide, setParams])

  const goTo = (target: string) => setParams({ date: target === today ? null : target, item: null })

  // When the details replace the list, move to them; beside the list, stay put.
  const showDetails = () => {
    if (!wide) requestAnimationFrame(() => detailHeading.current?.focus())
  }

  const select = (id: string) => {
    setParams({ item: id })
    showDetails()
  }

  const openLive = () => {
    if (live.status !== 'ready' || !live.data) return
    setParams({ date: null, item: live.data.id })
    showDetails()
  }

  const back = () => {
    const id = selectedId
    flushSync(() => setParams({ item: null }))
    // Return to the entry that was open, so keyboard users keep their place.
    if (id) document.querySelector<HTMLElement>(`[data-entry="${CSS.escape(id)}"]`)?.focus()
  }

  return (
    <>
      <a className="skip" href="#register">
        Skip to appointments
      </a>
      <p className="sample-banner">Sample data. This preview is not connected to the database yet.</p>

      <TopBar business={business} date={date} today={today} live={live} onDate={goTo} onOpenLive={openLive} />

      <main className="workspace" data-view={selection ? 'detail' : 'list'}>
        <section id="register" className="list-col" aria-label={`Appointments for ${formatDay(date)}`}>
          {day.status === 'loading' && <p className="loading">Loading appointments</p>}
          {day.status === 'error' && (
            <Notice
              tone="failed"
              title={`Could not load ${formatDay(date)}`}
              action={
                <button type="button" className="btn" onClick={retryDay}>
                  Try again
                </button>
              }
            >
              Check that the server is running, then try again.
            </Notice>
          )}
          {schedule && (
            <DayList
              schedule={schedule}
              isToday={date === today}
              timezone={business.timezone}
              selectedId={selection ? selectedId : null}
              onSelect={select}
            />
          )}
        </section>

        <section className="detail-col" aria-label="Call details">
          {selection ? (
            <CallPanel
              selection={selection}
              timezone={business.timezone}
              headingRef={detailHeading}
              onBack={back}
              onCancelBooking={(id) => setCancelled((previous) => new Set(previous).add(id))}
            />
          ) : schedule && nothingOnThisDay ? (
            <Notice title="Nothing to show for this day">Use Previous day or Next day to look at another day.</Notice>
          ) : schedule ? (
            <Notice title="Nothing selected">Select an appointment or a call on the left to see what was said.</Notice>
          ) : null}
        </section>
      </main>
    </>
  )
}
