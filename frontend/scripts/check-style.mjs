// Holds the source to the dashboard's visual rules, so a stray colour or shadow
// fails `npm run check` instead of slipping in unnoticed.
//
//   - only the five palette colours
//   - no gradients, glassmorphism (backdrop-filter), shadows or keyframe animation
//   - no capitalised labels, no emoji
//   - Atkinson Hyperlegible, and a prefers-reduced-motion rule

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, extname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const PALETTE = new Set(['#ffffff', '#1f3a5f', '#d9dee5', '#2e7d4f', '#b3261e'])
const COLOUR_NAME = /\b(white|black|gr[ae]y|silver|red|green|blue|navy|teal|orange|yellow|purple|pink|brown)\b/i
const COLOUR_PROPERTY = /(^|-)(color|background|border|outline|fill|stroke)($|-)/

function walk(directory) {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name)
    return statSync(path).isDirectory() ? walk(path) : [path]
  })
}

const files = [...walk(join(root, 'src')), join(root, 'index.html')].filter((path) =>
  ['.css', '.ts', '.tsx', '.html'].includes(extname(path)),
)

const problems = []
const report = (path, lineNumber, message) => problems.push(`${relative(root, path)}:${lineNumber}  ${message}`)

function expand(hex) {
  const h = hex.toLowerCase()
  return h.length === 4 ? `#${h[1]}${h[1]}${h[2]}${h[2]}${h[3]}${h[3]}` : h
}

let css = ''
for (const path of files) {
  const text = readFileSync(path, 'utf8')
  const isCss = extname(path) === '.css'
  if (isCss) css += text

  text.split('\n').forEach((line, index) => {
    const n = index + 1

    // Hex colours: anywhere in CSS, but in code only when written as a string.
    const hexes = isCss ? line.matchAll(/#[0-9a-fA-F]{3,8}\b/g) : line.matchAll(/(['"`])(#[0-9a-fA-F]{3,8})\1/g)
    for (const match of hexes) {
      const hex = expand(isCss ? match[0] : match[2])
      if (!PALETTE.has(hex)) report(path, n, `colour ${hex} is not in the palette`)
    }

    if (isCss) {
      for (const declaration of line.matchAll(/([a-z-]+)\s*:\s*([^;{}]+)/g)) {
        const value = declaration[2].replace(/var\([^)]*\)/g, '')
        if (COLOUR_PROPERTY.test(declaration[1]) && COLOUR_NAME.test(value)) {
          report(path, n, 'named colours are not allowed; use the palette variables')
        }
      }
    }
    if (/\b(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix)\(/i.test(line)) report(path, n, 'colour functions are not allowed; use the palette')
    if (/gradient\(/i.test(line)) report(path, n, 'gradients are not allowed')
    if (/backdrop-filter/i.test(line)) report(path, n, 'glassmorphism (backdrop-filter) is not allowed')
    if (/(box|text)-shadow|drop-shadow/i.test(line)) report(path, n, 'shadows are not allowed')
    if (/@keyframes/i.test(line)) report(path, n, 'keyframe animation is not allowed')
    if (/text-transform\s*:\s*(uppercase|capitalize)/i.test(line)) report(path, n, 'capitalised labels are not allowed')
    if (/\p{Extended_Pictographic}/u.test(line)) report(path, n, 'emoji are not allowed')
    if (/>\s*[A-Z][A-Z0-9 ]{3,}\s*</.test(line)) report(path, n, 'text written in capitals is not allowed')
  })
}

if (!/Atkinson Hyperlegible/.test(css)) problems.push('styles  the font family must be Atkinson Hyperlegible')
if (!/prefers-reduced-motion\s*:\s*reduce/.test(css)) problems.push('styles  there must be a prefers-reduced-motion rule')

if (problems.length > 0) {
  console.error(`Style rules broken (${problems.length}):\n  ${problems.join('\n  ')}`)
  process.exit(1)
}
console.log(`Style rules ok: ${files.length} files, palette only, no gradients, shadows, capitals or emoji.`)
