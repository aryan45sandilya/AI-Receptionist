import type { ReactNode } from 'react'

interface Props {
  title: string
  tone?: 'plain' | 'failed'
  action?: ReactNode
  children?: ReactNode
}

// Empty and error states: say what happened, then what to do about it.
export function Notice({ title, tone = 'plain', action, children }: Props) {
  return (
    <div className={`notice notice-${tone}`} role={tone === 'failed' ? 'alert' : undefined}>
      <p className="notice-title">{title}</p>
      {children && <p className="notice-body">{children}</p>}
      {action}
    </div>
  )
}
