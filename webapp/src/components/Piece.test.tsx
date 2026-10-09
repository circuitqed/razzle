import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import Piece from './Piece'

// Helper to render SVG components
function renderSvg(children: React.ReactNode) {
  return render(<svg>{children}</svg>)
}

describe('Piece component', () => {
  it('renders player 1 piece with correct color', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={false} />)
    const polygon = container.querySelector('polygon')
    expect(polygon).toHaveAttribute('fill', '#3b82f6') // Blue
  })

  it('renders player 2 piece with correct color', () => {
    const { container } = renderSvg(<Piece player={1} hasBall={false} />)
    const polygon = container.querySelector('polygon')
    expect(polygon).toHaveAttribute('fill', '#ef4444') // Red
  })

  it('shows ball indicator when hasBall is true', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={true} />)
    const circle = container.querySelector('circle')
    expect(circle).toBeInTheDocument()
    expect(circle).toHaveAttribute('fill', '#fbbf24') // Yellow/gold
  })

  it('does not show ball indicator when hasBall is false', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={false} />)
    const circle = container.querySelector('circle')
    expect(circle).not.toBeInTheDocument()
  })

  it('shows selection highlight when selected', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={false} isSelected={true} />)
    const polygon = container.querySelector('polygon')
    expect(polygon).toHaveAttribute('stroke', '#fbbf24') // Yellow highlight
    expect(polygon).toHaveAttribute('stroke-width', '3')
  })

  it('has normal stroke when not selected', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={false} isSelected={false} />)
    const polygon = container.querySelector('polygon')
    expect(polygon).toHaveAttribute('stroke', '#1f2937')
    expect(polygon).toHaveAttribute('stroke-width', '1.5')
  })

  it('draws an ineligible piece hatched, with no facet', () => {
    const { container } = renderSvg(<Piece player={0} hasBall={false} isIneligible={true} />)
    const polygons = container.querySelectorAll('polygon')
    expect(polygons).toHaveLength(1)
    expect(polygons[0]).toHaveAttribute('fill', 'url(#kb-hatch-0)')
    expect(container.querySelector('pattern#kb-hatch-0')).toBeInTheDocument()
  })

  it('draws an eligible piece solid with a facet', () => {
    const { container } = renderSvg(<Piece player={1} hasBall={false} isIneligible={false} />)
    const polygons = container.querySelectorAll('polygon')
    expect(polygons[0]).toHaveAttribute('fill', '#ef4444')
    expect(polygons[1]).toHaveAttribute('fill', '#ffffff')
  })
})
