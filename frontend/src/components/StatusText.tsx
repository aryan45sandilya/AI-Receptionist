import type { Tone } from '../labels'

export function StatusText({ label, tone }: { label: string; tone: Tone }) {
  return <span className={`status status-${tone}`}>{label}</span>
}
