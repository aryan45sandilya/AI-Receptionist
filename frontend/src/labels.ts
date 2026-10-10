// Plain-language wording for the values the database stores.

import type { BookingSource, BookingStatus, CallChannel, CallStatus, ToolName } from './types'

// "booked" is green, "failed" is red, everything else is plain ink. The word
// always carries the meaning; colour only backs it up.
export type Tone = 'booked' | 'failed' | 'plain'

export const BOOKING_STATUS: Record<BookingStatus, { label: string; tone: Tone }> = {
  booked: { label: 'Booked', tone: 'booked' },
  completed: { label: 'Completed', tone: 'plain' },
  cancelled: { label: 'Cancelled', tone: 'plain' },
  no_show: { label: 'No-show', tone: 'failed' },
}

export const CALL_STATUS: Record<CallStatus, { label: string; tone: Tone }> = {
  in_progress: { label: 'Live', tone: 'booked' },
  completed: { label: 'No booking', tone: 'plain' },
  handed_off: { label: 'Handed to a person', tone: 'plain' },
  failed: { label: 'Failed', tone: 'failed' },
}

export const CHANNEL: Record<CallChannel, string> = {
  web_demo: 'Browser call',
  twilio_voice: 'Phone call',
  whatsapp: 'WhatsApp',
  terminal: 'Text chat',
}

export const SOURCE: Record<BookingSource, string> = {
  voice: 'Phone call',
  whatsapp: 'WhatsApp',
  dashboard: 'Front desk',
}

// What the agent did between two sentences, said the way the owner would.
export const TOOL_NOTE: Record<ToolName, string> = {
  check_availability: 'Checked the calendar',
  book_appointment: 'Booked the appointment',
  cancel_appointment: 'Cancelled the appointment',
  reschedule_appointment: 'Moved the appointment',
  handoff_to_human: 'Put the caller through to a person',
}

export function isVoice(channel: CallChannel): boolean {
  return channel === 'web_demo' || channel === 'twilio_voice'
}
