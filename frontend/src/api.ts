// The one place the dashboard gets its data. For now it reads the sample data;
// when the backend has these endpoints, each function becomes a fetch and nothing
// else in the app changes.
//
// To see how errors look, add ?fail=day or ?fail=call to the address.

import { BUSINESS, sampleCall, sampleDay, sampleLiveCall } from './sample'
import type { Business, Call, CallSummary, DaySchedule } from './types'

function shouldFail(what: 'day' | 'call'): boolean {
  return new URLSearchParams(window.location.search).get('fail') === what
}

export const api = {
  business: BUSINESS satisfies Business,

  async loadDay(date: string): Promise<DaySchedule> {
    if (shouldFail('day')) throw new Error('Sample failure')
    return sampleDay(date)
  },

  async loadCall(id: string): Promise<Call> {
    if (shouldFail('call')) throw new Error('Sample failure')
    const call = sampleCall(id)
    if (!call) throw new Error(`No call ${id}`)
    return call
  },

  async loadLiveCall(): Promise<CallSummary | null> {
    return sampleLiveCall()
  },
}
