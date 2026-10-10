// Sample data that stands in for the API until it exists. The business, services,
// doctors and opening hours are the ones scripts/seed_demo.py creates, so this
// matches the real demo business. None of it is read from the database.
//
// Everything is anchored to the moment the page loaded: "today" has appointments,
// times that have passed show as completed, later ones as booked, and one call is
// live, so the screen looks the same whenever it is opened.

import type {
  Booking,
  BookingSource,
  BookingStatus,
  Business,
  Call,
  CallChannel,
  CallStatus,
  DaySchedule,
  ReplyTimings,
  ToolName,
  Turn,
  TurnRole,
} from './types'
import { dateIn, daysBetween, formatTime, weekdayName, weekdayOf } from './time'

export const BUSINESS: Business = {
  name: 'Sunrise Dental',
  timezone: 'Asia/Kolkata',
  closedWeekdays: [0], // Sunday
}

const GREETING = 'Sunrise Dental, how can I help?'
// India has no daylight saving, so a fixed offset is exact here. Rows from the
// real API carry their own offsets.
const OFFSET = '+05:30'
const LOADED_AT = Date.now()

type Service = 'Consultation' | 'Dental cleaning' | 'Root canal'
type Doctor = 'Dr Rao' | 'Dr Menon'

const MINUTES: Record<Service, number> = {
  Consultation: 15,
  'Dental cleaning': 30,
  'Root canal': 60,
}

interface Seed {
  time: string // local HH:MM
  name: string
  service: Service
  staff: Doctor | null
  source: BookingSource
  fate?: 'cancelled' | 'no_show'
  opening?: string // how the caller first asked, when it is worth showing
}

// Times sit inside Mon-Sat 09:00-13:00 and 17:00-20:00 and never overlap for the
// same doctor, as the real schedule would not allow.
const TODAY: Seed[] = [
  { time: '09:00', name: 'Anita Sharma', service: 'Consultation', staff: 'Dr Rao', source: 'voice' },
  { time: '09:15', name: 'Rahul Verma', service: 'Consultation', staff: 'Dr Menon', source: 'voice', fate: 'cancelled' },
  { time: '09:30', name: 'Meera Iyer', service: 'Dental cleaning', staff: 'Dr Rao', source: 'whatsapp' },
  {
    time: '10:00', name: 'Karthik Subramanian', service: 'Root canal', staff: 'Dr Menon', source: 'voice',
    opening: 'Hello, enakku oru root canal appointment venum.',
  },
  { time: '10:30', name: 'Imran Qureshi', service: 'Dental cleaning', staff: 'Dr Rao', source: 'voice', fate: 'no_show' },
  { time: '11:30', name: 'Divya Reddy', service: 'Consultation', staff: null, source: 'dashboard' },
  { time: '12:00', name: 'Arjun Mehta', service: 'Root canal', staff: 'Dr Rao', source: 'voice' },
  { time: '17:00', name: 'Sneha Patil', service: 'Dental cleaning', staff: 'Dr Menon', source: 'voice' },
  { time: '17:30', name: 'Vikram Singh', service: 'Consultation', staff: 'Dr Rao', source: 'whatsapp' },
  { time: '18:30', name: 'Lakshmi Narayanan', service: 'Dental cleaning', staff: 'Dr Rao', source: 'voice' },
  { time: '19:00', name: 'Farah Khan', service: 'Consultation', staff: 'Dr Menon', source: 'voice' },
]

// Keyed by days from today.
const OTHER_DAYS: Record<number, Seed[]> = {
  [-2]: [
    { time: '09:30', name: 'Naveen Gupta', service: 'Consultation', staff: 'Dr Menon', source: 'voice' },
    { time: '10:00', name: 'Pooja Bansal', service: 'Dental cleaning', staff: 'Dr Rao', source: 'whatsapp' },
    { time: '11:00', name: 'Suresh Kumar', service: 'Root canal', staff: 'Dr Menon', source: 'voice', fate: 'no_show' },
    { time: '17:30', name: 'Kavya Menon', service: 'Dental cleaning', staff: 'Dr Rao', source: 'voice' },
    { time: '18:15', name: 'Rohan Das', service: 'Consultation', staff: 'Dr Menon', source: 'voice' },
  ],
  [-1]: [
    { time: '09:00', name: 'Isha Kulkarni', service: 'Dental cleaning', staff: 'Dr Rao', source: 'voice' },
    { time: '12:00', name: 'Gaurav Saxena', service: 'Consultation', staff: 'Dr Menon', source: 'voice' },
    { time: '17:00', name: 'Latha Krishnan', service: 'Root canal', staff: 'Dr Rao', source: 'voice' },
  ],
  1: [
    { time: '09:00', name: 'Nisha Thomas', service: 'Dental cleaning', staff: 'Dr Rao', source: 'voice' },
    { time: '10:30', name: 'Aditya Joshi', service: 'Consultation', staff: 'Dr Menon', source: 'whatsapp' },
    { time: '12:00', name: 'Gayatri Pillai', service: 'Root canal', staff: 'Dr Rao', source: 'voice' },
    { time: '17:00', name: 'Harish Babu', service: 'Dental cleaning', staff: 'Dr Menon', source: 'voice', fate: 'cancelled' },
    { time: '18:00', name: 'Tanya Kapoor', service: 'Consultation', staff: 'Dr Rao', source: 'voice' },
  ],
  2: [
    { time: '09:00', name: 'Manoj Nair', service: 'Consultation', staff: 'Dr Rao', source: 'voice' },
    { time: '11:00', name: 'Shreya Banerjee', service: 'Dental cleaning', staff: 'Dr Menon', source: 'whatsapp' },
    { time: '17:30', name: 'Yusuf Ansari', service: 'Root canal', staff: 'Dr Rao', source: 'voice' },
  ],
  3: [
    { time: '10:00', name: 'Ritu Malhotra', service: 'Dental cleaning', staff: 'Dr Menon', source: 'voice' },
    { time: '18:30', name: 'Deepak Chawla', service: 'Consultation', staff: 'Dr Rao', source: 'voice' },
  ],
}

// ---------------------------------------------------------------------------
// Small deterministic helpers: the same name always gets the same phone number
// and the same reference, so reloading the page does not reshuffle the data.

function hash(text: string): number {
  let h = 2166136261
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function random(seed: string): () => number {
  let a = hash(seed)
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const REFERENCE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789' // no 0, O, 1 or I, like the real ones

function referenceFor(key: string): string {
  const next = random(`reference-${key}`)
  return Array.from({ length: 6 }, () => REFERENCE_ALPHABET[Math.floor(next() * REFERENCE_ALPHABET.length)]).join('')
}

function phoneFor(key: string): string {
  return `+9180000${String(hash(key) % 100000).padStart(5, '0')}`
}

function startOfDay(date: string): number {
  return Date.parse(`${date}T00:00:00${OFFSET}`)
}

// ---------------------------------------------------------------------------
// Latency numbers in the ranges measured on the real pipeline (Phase 3).

function voiceReply(next: () => number, lookup: boolean): ReplyTimings {
  const sttFinalMs = Math.round(840 + next() * 120)
  if (!lookup) {
    const llmFirstTokenMs = Math.round(sttFinalMs + 700 + next() * 500)
    const totalMs = Math.round(llmFirstTokenMs + 150 + next() * 250)
    return { sttFinalMs, llmFirstTokenMs, ttsFirstAudioMs: totalMs, totalMs, fillerMs: null }
  }
  const llmFirstTokenMs = Math.round(sttFinalMs + 1100 + next() * 500)
  const fillerMs = Math.round(llmFirstTokenMs + 150 + next() * 250)
  const totalMs = Math.round(fillerMs + 1500 + next() * 1800)
  return { sttFinalMs, llmFirstTokenMs, ttsFirstAudioMs: totalMs, totalMs, fillerMs }
}

function voiceGreeting(next: () => number): ReplyTimings {
  const first = Math.round(280 + next() * 80)
  return { sttFinalMs: null, llmFirstTokenMs: null, ttsFirstAudioMs: first, totalMs: first, fillerMs: null }
}

// On a text channel there is no audio: the one number is "reply complete".
function textReply(next: () => number): ReplyTimings {
  return {
    sttFinalMs: null,
    llmFirstTokenMs: null,
    ttsFirstAudioMs: null,
    totalMs: Math.round(1800 + next() * 1800),
    fillerMs: null,
  }
}

// ---------------------------------------------------------------------------
// Conversations

interface Step {
  role: TurnRole
  text: string
  after: number // seconds since the previous step
  tools?: ToolName[]
  bargeIn?: boolean
  timings?: ReplyTimings
}

function turnsFrom(steps: Step[], startedAt: number): Turn[] {
  let moment = startedAt
  return steps.map((step, index) => {
    moment += step.after * 1000
    return {
      index,
      role: step.role,
      text: step.text,
      at: new Date(moment).toISOString(),
      bargeIn: step.bargeIn ?? false,
      tools: step.tools ?? [],
      timings: step.role === 'agent' ? (step.timings ?? null) : null,
    }
  })
}

interface CallDetails {
  id: string
  startedAt: number
  channel: CallChannel
  fromPhone: string
  status: CallStatus
  steps: Step[]
  handoffReason?: string
  summary?: string
  durationMs?: number | null
}

function makeCall(details: CallDetails): Call {
  const turns = turnsFrom(details.steps, details.startedAt)
  const last = turns[turns.length - 1]
  const lastAt = last ? Date.parse(last.at) : details.startedAt
  return {
    id: details.id,
    channel: details.channel,
    fromPhone: details.fromPhone,
    status: details.status,
    handoffReason: details.handoffReason ?? null,
    summary: details.summary ?? null,
    startedAt: new Date(details.startedAt).toISOString(),
    durationMs:
      details.durationMs !== undefined
        ? details.durationMs
        : details.status === 'in_progress'
          ? null
          : lastAt - details.startedAt + 3000,
    turns,
  }
}

function bookingSteps(booking: Booking, seed: Seed, next: () => number): Step[] {
  const voice = seed.source === 'voice'
  const reply = (lookup: boolean) => (voice ? voiceReply(next, lookup) : textReply(next))
  const day = weekdayName(dateIn(BUSINESS.timezone, booking.slotStart))
  const time = formatTime(booking.slotStart, BUSINESS.timezone)
  const withDoctor = booking.staffName ? ` with ${booking.staffName}` : ''
  const part = Number(seed.time.slice(0, 2)) < 13 ? 'morning' : 'evening'
  const gap = voice ? 1 : 3 // people take longer to type than to speak

  return [
    { role: 'agent', after: 0, text: GREETING, timings: voice ? voiceGreeting(next) : undefined },
    {
      role: 'caller',
      after: 6 * gap,
      text: seed.opening ?? `Hi, I'd like to book a ${booking.serviceName.toLowerCase()}.`,
    },
    { role: 'agent', after: 3, text: 'Sure. Which day suits you?', timings: reply(false) },
    { role: 'caller', after: 5 * gap, text: `${day}, ${part} if possible.` },
    {
      role: 'agent',
      after: 4,
      tools: ['check_availability'],
      text: `I have ${time} on ${day}${withDoctor}. Shall I book it?`,
      timings: reply(true),
    },
    { role: 'caller', after: 5 * gap, text: `Yes, please. The name is ${booking.customerName}.` },
    {
      role: 'agent',
      after: 4,
      tools: ['book_appointment'],
      text: `Done. ${booking.serviceName} on ${day} at ${time}${withDoctor}. Your reference is ${booking.reference}.`,
      timings: reply(true),
    },
    { role: 'caller', after: 4 * gap, text: 'Thank you.' },
    { role: 'agent', after: 2, text: "You're welcome. See you then. Goodbye.", timings: reply(false) },
  ]
}

function bookingCall(booking: Booking, seed: Seed): Call {
  const next = random(`call-${booking.id}`)
  // The call that made the booking came before the appointment, and not from the future.
  const lead = (3 + next() * 49) * 3_600_000
  const slot = Date.parse(booking.slotStart)
  const startedAt = Math.min(slot - lead, LOADED_AT - 600_000 - next() * 3_600_000)
  const channel: CallChannel = seed.source === 'whatsapp' ? 'whatsapp' : 'twilio_voice'
  return makeCall({
    id: booking.callId ?? `call-${booking.id}`,
    startedAt,
    channel,
    fromPhone: booking.customerPhone,
    status: 'completed',
    steps: bookingSteps(booking, seed, next),
    summary: `Booked ${booking.serviceName.toLowerCase()} for ${weekdayName(dateIn(BUSINESS.timezone, booking.slotStart))} at ${formatTime(booking.slotStart, BUSINESS.timezone)}.`,
  })
}

function makeBooking(date: string, seed: Seed): Booking {
  const start = Date.parse(`${date}T${seed.time}:00${OFFSET}`)
  const end = start + MINUTES[seed.service] * 60_000
  const id = `bk-${date}-${seed.time.replace(':', '')}`
  let status: BookingStatus = 'booked'
  if (seed.fate === 'cancelled') status = 'cancelled'
  else if (end < LOADED_AT) status = seed.fate === 'no_show' ? 'no_show' : 'completed'
  return {
    id,
    reference: referenceFor(id),
    customerName: seed.name,
    customerPhone: phoneFor(seed.name),
    serviceName: seed.service,
    staffName: seed.staff,
    slotStart: new Date(start).toISOString(),
    slotEnd: new Date(end).toISOString(),
    status,
    source: seed.source,
    // A booking typed in at the front desk has no call behind it.
    callId: seed.source === 'dashboard' ? null : `call-${id}`,
  }
}

// ---------------------------------------------------------------------------
// Calls that did not end in a booking

// A moment on `date`, `minutesAgo` before the page loaded but never before `floor`
// minutes past midnight and never in the future.
function earlierToday(date: string, minutesAgo: number, floor: number): number {
  const wanted = Math.max(startOfDay(date) + floor * 60_000, LOADED_AT - minutesAgo * 60_000)
  return Math.min(wanted, LOADED_AT - 120_000)
}

function liveCall(): Call {
  const next = random('live')
  return makeCall({
    id: 'call-live',
    startedAt: LOADED_AT - 30_000,
    channel: 'twilio_voice',
    fromPhone: phoneFor('live caller'),
    status: 'in_progress',
    // The last steps are still in the future when the page opens, so they arrive
    // while it is open, the way words would from a call in progress.
    steps: [
      { role: 'agent', after: 0, text: GREETING, timings: voiceGreeting(next) },
      { role: 'caller', after: 8, text: 'Hi, what time do you open on Saturdays?' },
      {
        role: 'agent',
        after: 6,
        text: 'On Saturdays we are open from 9 am to 1 pm, and again from 5 pm to 8 pm.',
        timings: voiceReply(next, false),
      },
      { role: 'caller', after: 13, text: 'Great. And do you do root canals?' },
      {
        role: 'agent',
        after: 6,
        text: 'Yes, a root canal takes about an hour. Would you like to book one?',
        timings: voiceReply(next, false),
      },
      { role: 'caller', after: 12, text: 'Yes, please. Saturday morning if you have anything.' },
      {
        role: 'agent',
        after: 8,
        tools: ['check_availability'],
        text: 'Let me check Saturday morning for you.',
        timings: voiceReply(next, true),
      },
    ],
  })
}

function handoffCall(date: string, startedAt: number): Call {
  const next = random(`handoff-${date}`)
  return makeCall({
    id: `call-handoff-${date}`,
    startedAt,
    channel: 'twilio_voice',
    fromPhone: phoneFor('upset caller'),
    status: 'handed_off',
    handoffReason: 'The caller disputed a charge and asked for a person twice.',
    summary: 'Says they were charged for a cleaning they did not have.',
    steps: [
      { role: 'agent', after: 0, text: GREETING, timings: voiceGreeting(next) },
      { role: 'caller', after: 7, text: 'I was charged for a cleaning I never had. I want to talk to someone.' },
      {
        role: 'agent',
        after: 3,
        text: "I'm sorry about that. I can't see billing, but I can",
        bargeIn: true,
        timings: voiceReply(next, false),
      },
      { role: 'caller', after: 4, text: 'No. Just get me a person.' },
      {
        role: 'agent',
        after: 4,
        tools: ['handoff_to_human'],
        text: "Of course. I'm putting you through to the front desk now.",
        // A handoff is never preceded by a stopgap phrase, so no filler time.
        timings: voiceReply(next, false),
      },
    ],
  })
}

function droppedCall(date: string, startedAt: number): Call {
  const next = random(`dropped-${date}`)
  const heard = voiceReply(next, false)
  return makeCall({
    id: `call-dropped-${date}`,
    startedAt,
    channel: 'twilio_voice',
    fromPhone: phoneFor(`dropped ${date}`),
    status: 'failed',
    summary: 'The line dropped after 14 seconds. Nothing was booked.',
    durationMs: 14_000,
    steps: [
      { role: 'agent', after: 0, text: GREETING, timings: voiceGreeting(next) },
      { role: 'caller', after: 6, text: 'Hello? Can you hear me?' },
      {
        role: 'agent',
        after: 3,
        text: 'Yes, I can hear you. How can I help?',
        // The model answered but the voice never reached the caller: the turn's
        // record stops at the first word, and there is no time-to-first-sound.
        timings: { ...heard, ttsFirstAudioMs: null, totalMs: null },
      },
    ],
  })
}

function askedCall(date: string, startedAt: number, question: string, answer: string, summary: string): Call {
  const next = random(`asked-${date}`)
  return makeCall({
    id: `call-asked-${date}`,
    startedAt,
    channel: 'twilio_voice',
    fromPhone: phoneFor(`asked ${date}`),
    status: 'completed',
    summary,
    steps: [
      { role: 'agent', after: 0, text: GREETING, timings: voiceGreeting(next) },
      { role: 'caller', after: 6, text: question },
      { role: 'agent', after: 3, text: answer, timings: voiceReply(next, false) },
      { role: 'caller', after: 6, text: 'Okay, thank you.' },
      { role: 'agent', after: 2, text: 'You are welcome. Goodbye.', timings: voiceReply(next, false) },
    ],
  })
}

function otherCallsFor(date: string, today: string): Call[] {
  const offset = daysBetween(today, date)
  if (offset === 0) {
    return [
      liveCall(),
      handoffCall(date, earlierToday(date, 55, 12)),
      droppedCall(date, earlierToday(date, 130, 6)),
      askedCall(
        date,
        earlierToday(date, 210, 3),
        'How long does a root canal take?',
        'A root canal takes about an hour. Would you like to book one?',
        'Asked how long a root canal takes. Did not book.',
      ),
    ]
  }
  if (offset === -1) {
    return [
      askedCall(
        date,
        startOfDay(date) + 11 * 3_600_000 + 42 * 60_000,
        'Are you open on Sunday?',
        'We are closed on Sundays. We open again on Monday at 9 am.',
        'Asked whether the clinic is open on Sunday. Told it is closed.',
      ),
    ]
  }
  if (offset === -2) return [droppedCall(date, startOfDay(date) + 12 * 3_600_000 + 20 * 60_000)]
  return []
}

// ---------------------------------------------------------------------------

interface DayData {
  schedule: DaySchedule
  calls: Map<string, Call>
}

const days = new Map<string, DayData>()

function dayData(date: string): DayData {
  const cached = days.get(date)
  if (cached) return cached

  const today = dateIn(BUSINESS.timezone, LOADED_AT)
  const open = !BUSINESS.closedWeekdays.includes(weekdayOf(date))
  const seeds = !open ? [] : date === today ? TODAY : (OTHER_DAYS[daysBetween(today, date)] ?? [])

  const calls = new Map<string, Call>()
  const bookings = seeds.map((seed) => {
    const booking = makeBooking(date, seed)
    if (booking.callId) calls.set(booking.callId, bookingCall(booking, seed))
    return booking
  })

  const others = otherCallsFor(date, today)
  for (const call of others) calls.set(call.id, call)

  const data: DayData = {
    schedule: {
      date,
      open,
      bookings,
      otherCalls: others.map((call) => ({
        id: call.id,
        channel: call.channel,
        fromPhone: call.fromPhone,
        status: call.status,
        handoffReason: call.handoffReason,
        summary: call.summary,
        startedAt: call.startedAt,
        durationMs: call.durationMs,
      })),
    },
    calls,
  }
  days.set(date, data)
  return data
}

export function sampleDay(date: string): DaySchedule {
  return dayData(date).schedule
}

// Booking calls are filed under the date of the appointment they made, and the
// others under the day they happened, and both ids carry that date.
export function sampleCall(id: string): Call | undefined {
  if (id === 'call-live') return dayData(dateIn(BUSINESS.timezone, LOADED_AT)).calls.get(id)
  const date = id.match(/\d{4}-\d{2}-\d{2}/)?.[0]
  return date ? dayData(date).calls.get(id) : undefined
}

export function sampleLiveCall(): Call | null {
  return sampleCall('call-live') ?? null
}
