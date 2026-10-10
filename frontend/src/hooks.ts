import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

export type Async<T> = { status: 'loading' } | { status: 'ready'; data: T } | { status: 'error' }

const LOADING = { status: 'loading' } as const

/**
 * Runs `load` and again whenever `key` changes, and returns where it has got to
 * plus a function that tries again. A result is only shown for the key it was
 * asked for, so changing the day never shows the old day under the new heading.
 */
export function useAsync<T>(load: () => Promise<T>, key: string): [Async<T>, () => void] {
  const [attempt, setAttempt] = useState(0)
  const [settled, setSettled] = useState<{ id: string; value: Async<T> } | null>(null)
  const id = `${key}#${attempt}`

  useEffect(() => {
    let current = true
    load().then(
      (data) => current && setSettled({ id, value: { status: 'ready', data } }),
      () => current && setSettled({ id, value: { status: 'error' } }),
    )
    return () => {
      current = false
    }
    // `load` is a new function on every render; the id says when its answer would differ.
  }, [id])

  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  return [settled?.id === id ? settled.value : LOADING, retry]
}

/** The current time, refreshed every `intervalMs`; pass null to stop the clock. */
export function useNow(intervalMs: number | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (intervalMs === null) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

type Changes = Record<string, string | null>

/**
 * The address bar holds which day and which item are open, so a link reproduces
 * the screen and the browser's Back button steps through what was looked at.
 */
export function useUrlState(): [URLSearchParams, (changes: Changes, mode?: 'push' | 'replace') => void] {
  const [params, setParams] = useState(() => new URLSearchParams(window.location.search))

  useEffect(() => {
    const sync = () => setParams(new URLSearchParams(window.location.search))
    window.addEventListener('popstate', sync)
    return () => window.removeEventListener('popstate', sync)
  }, [])

  const update = useCallback((changes: Changes, mode: 'push' | 'replace' = 'push') => {
    const next = new URLSearchParams(window.location.search)
    for (const [name, value] of Object.entries(changes)) {
      if (value === null) next.delete(name)
      else next.set(name, value)
    }
    const query = next.toString()
    const url = `${window.location.pathname}${query ? `?${query}` : ''}`
    if (mode === 'push') window.history.pushState(null, '', url)
    else window.history.replaceState(null, '', url)
    setParams(next)
  }, [])

  return [params, update]
}

const WIDE = '(min-width: 60rem)'

/** True when the list and the details sit side by side rather than one at a time. */
export function useWide(): boolean {
  return useSyncExternalStore(
    (notify) => {
      const query = window.matchMedia(WIDE)
      query.addEventListener('change', notify)
      return () => query.removeEventListener('change', notify)
    },
    () => window.matchMedia(WIDE).matches,
  )
}
