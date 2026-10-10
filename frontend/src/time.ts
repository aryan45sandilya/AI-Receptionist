// Everything here is read in the business's own time zone, never the browser's:
// an owner travelling, or a laptop set to another zone, must still see "9:30 am"
// for a 9:30 am appointment.

const DAY_MS = 86_400_000

function parts(date: string): [number, number, number] {
  const [year, month, day] = date.split('-').map(Number)
  return [year, month, day]
}

function utcMidnight(date: string): number {
  const [year, month, day] = parts(date)
  return Date.UTC(year, month - 1, day)
}

export function isDate(value: string | null): value is string {
  return value !== null && /^\d{4}-\d{2}-\d{2}$/.test(value) && !Number.isNaN(utcMidnight(value))
}

/** The calendar date, YYYY-MM-DD, at `instant` in `timeZone`. */
export function dateIn(timeZone: string, instant: Date | string | number = Date.now()): string {
  // The en-CA locale writes dates as YYYY-MM-DD.
  return new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(new Date(instant))
}

export function addDays(date: string, days: number): string {
  return new Date(utcMidnight(date) + days * DAY_MS).toISOString().slice(0, 10)
}

export function daysBetween(from: string, to: string): number {
  return Math.round((utcMidnight(to) - utcMidnight(from)) / DAY_MS)
}

/** 0 = Sunday ... 6 = Saturday. */
export function weekdayOf(date: string): number {
  return new Date(utcMidnight(date)).getUTCDay()
}

function words(date: string, options: Intl.DateTimeFormatOptions): Map<string, string> {
  const formatted = new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', ...options })
    .formatToParts(new Date(utcMidnight(date)))
  return new Map(formatted.map((part) => [part.type, part.value]))
}

/** "Monday 5 October" */
export function formatDay(date: string): string {
  const p = words(date, { weekday: 'long', day: 'numeric', month: 'long' })
  return `${p.get('weekday')} ${p.get('day')} ${p.get('month')}`
}

export function weekdayName(date: string): string {
  return words(date, { weekday: 'long' }).get('weekday') ?? ''
}

/** "9:30 am" */
export function formatTime(instant: string, timeZone: string): string {
  return new Intl.DateTimeFormat('en-IN', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }).format(new Date(instant))
}

/** "Sun 4 Oct, 4:12 pm" */
export function formatMoment(instant: string, timeZone: string): string {
  const day = new Intl.DateTimeFormat('en-GB', {
    timeZone,
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  }).formatToParts(new Date(instant))
  const p = new Map(day.map((part) => [part.type, part.value]))
  return `${p.get('weekday')} ${p.get('day')} ${p.get('month')}, ${formatTime(instant, timeZone)}`
}

/** "2:05" for 125 000 ms: how far into a call something happened. */
export function formatClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const seconds = String(total % 60).padStart(2, '0')
  return `${Math.floor(total / 60)}:${seconds}`
}

/** "2 min 14 s" */
export function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000)
  if (total < 60) return `${total} s`
  return `${Math.floor(total / 60)} min ${String(total % 60).padStart(2, '0')} s`
}

/** "2.1 s" */
export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`
}

/** "+91 80000 00101" for an Indian number; anything else is shown as given. */
export function formatPhone(e164: string): string {
  if (/^\+91\d{10}$/.test(e164)) return `+91 ${e164.slice(3, 8)} ${e164.slice(8)}`
  return e164
}
