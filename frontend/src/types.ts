// The shapes the dashboard works with. They follow backend/app/models.py and
// backend/app/enums.py, so the API can hand rows over without a translation step.
// Timestamps are ISO 8601 strings that carry their offset.

export type BookingStatus = 'booked' | 'cancelled' | 'completed' | 'no_show'
export type BookingSource = 'voice' | 'whatsapp' | 'dashboard'
export type CallChannel = 'web_demo' | 'twilio_voice' | 'whatsapp' | 'terminal'
export type CallStatus = 'in_progress' | 'completed' | 'failed' | 'handed_off'
export type TurnRole = 'caller' | 'agent'
export type ToolName =
  | 'check_availability'
  | 'book_appointment'
  | 'cancel_appointment'
  | 'reschedule_appointment'
  | 'handoff_to_human'

export interface Business {
  name: string
  timezone: string // IANA name, e.g. "Asia/Kolkata"
  closedWeekdays: number[] // 0 = Sunday ... 6 = Saturday
}

export interface Booking {
  id: string
  reference: string
  customerName: string
  customerPhone: string
  serviceName: string
  staffName: string | null
  slotStart: string
  slotEnd: string
  status: BookingStatus
  source: BookingSource
  callId: string | null
}

export interface CallSummary {
  id: string
  channel: CallChannel
  fromPhone: string | null
  status: CallStatus
  handoffReason: string | null
  summary: string | null
  startedAt: string
  durationMs: number | null
}

// Milliseconds after the caller stopped speaking (call_turns, agent rows).
// Any of them can be null: a turn that failed part-way records how far it got.
export interface ReplyTimings {
  sttFinalMs: number | null
  llmFirstTokenMs: number | null
  ttsFirstAudioMs: number | null
  totalMs: number | null
  // When the line stopped being silent, if that was a stopgap ("one moment")
  // said while a lookup ran. Null when the first sound was the answer itself.
  fillerMs: number | null
}

export interface Turn {
  index: number
  role: TurnRole
  text: string
  at: string
  bargeIn: boolean // the caller spoke over this reply
  tools: ToolName[]
  timings: ReplyTimings | null
}

export interface Call extends CallSummary {
  turns: Turn[]
}

export interface DaySchedule {
  date: string // YYYY-MM-DD in the business's time zone
  open: boolean // false on a weekday the business never opens
  bookings: Booking[]
  // Calls that began on this day and did not end in a booking.
  otherCalls: CallSummary[]
}
