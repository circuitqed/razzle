import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, fireEvent } from '@testing-library/react'
import { useState } from 'react'
import { useDialogA11y } from './useDialogA11y'

beforeAll(() => {
  // jsdom has no layout; treat every element as visible.
  Element.prototype.getClientRects = function () { return [{}] as unknown as DOMRectList }
})

function Dialog({ onClose }: { onClose?: () => void }) {
  const ref = useDialogA11y(onClose)
  return (
    <div ref={ref} role="dialog" data-testid="dialog">
      <button>first</button>
      <button>last</button>
    </div>
  )
}

function Harness({ onClose }: { onClose?: () => void }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button onClick={() => setOpen(true)}>opener</button>
      {open && <Dialog onClose={() => { onClose?.(); setOpen(false) }} />}
    </>
  )
}

describe('useDialogA11y', () => {
  it('moves focus in, traps Tab, closes on Escape and restores focus', () => {
    const onClose = vi.fn()
    const { getByText, getByTestId, queryByTestId } = render(<Harness onClose={onClose} />)
    const opener = getByText('opener')
    opener.focus()
    fireEvent.click(opener)

    const dialog = getByTestId('dialog')
    expect(document.activeElement).toBe(dialog)

    // Shift+Tab from the container wraps to the last control
    fireEvent.keyDown(document, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(getByText('last'))
    // Tab from the last wraps to the first
    fireEvent.keyDown(document, { key: 'Tab' })
    expect(document.activeElement).toBe(getByText('first'))

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(queryByTestId('dialog')).toBeNull()
    expect(document.activeElement).toBe(opener)
  })

  it('ignores Escape when no onClose is given', () => {
    const { getByTestId } = render(<Dialog />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(getByTestId('dialog')).toBeTruthy()
  })

  it('only the topmost dialog handles keys', () => {
    const outer = vi.fn()
    const inner = vi.fn()
    render(<><Dialog onClose={outer} /><Dialog onClose={inner} /></>)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(inner).toHaveBeenCalledTimes(1)
    expect(outer).not.toHaveBeenCalled()
  })
})
