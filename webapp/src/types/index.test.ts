import { describe, it, expect } from 'vitest'
import {
  encodeMove,
  decodeMove,
  squareToCoords,
  coordsToSquare,
  hasPiece,
  getPiecePositions,
  squareToAlgebraic,
  algebraicToSquare,
  BOARD_COLS,
  BOARD_ROWS,
  TOTAL_SQUARES,
} from './index'

describe('Board constants', () => {
  it('should have correct dimensions', () => {
    expect(BOARD_COLS).toBe(7)
    expect(BOARD_ROWS).toBe(8)
    expect(TOTAL_SQUARES).toBe(56)
  })
})

describe('Move encoding/decoding', () => {
  it('should encode a move correctly', () => {
    // d1 (square 3) to c3 (square 16) = 3 * 56 + 16 = 184
    expect(encodeMove(3, 16)).toBe(184)
  })

  it('should decode a move correctly', () => {
    const { src, dst } = decodeMove(184)
    expect(src).toBe(3)
    expect(dst).toBe(16)
  })

  it('should be reversible', () => {
    const src = 10
    const dst = 25
    const encoded = encodeMove(src, dst)
    const decoded = decodeMove(encoded)
    expect(decoded.src).toBe(src)
    expect(decoded.dst).toBe(dst)
  })
})

describe('Square coordinates', () => {
  it('should convert square to coords', () => {
    // Square 0 = a1 = (row 0, col 0)
    expect(squareToCoords(0)).toEqual({ row: 0, col: 0 })
    // Square 3 = d1 = (row 0, col 3)
    expect(squareToCoords(3)).toEqual({ row: 0, col: 3 })
    // Square 7 = a2 = (row 1, col 0)
    expect(squareToCoords(7)).toEqual({ row: 1, col: 0 })
    // Square 55 = g8 = (row 7, col 6)
    expect(squareToCoords(55)).toEqual({ row: 7, col: 6 })
  })

  it('should convert coords to square', () => {
    expect(coordsToSquare(0, 0)).toBe(0)
    expect(coordsToSquare(0, 3)).toBe(3)
    expect(coordsToSquare(1, 0)).toBe(7)
    expect(coordsToSquare(7, 6)).toBe(55)
  })

  it('should be reversible', () => {
    for (let sq = 0; sq < TOTAL_SQUARES; sq++) {
      const { row, col } = squareToCoords(sq)
      expect(coordsToSquare(row, col)).toBe(sq)
    }
  })
})

describe('Algebraic notation', () => {
  it('should convert square to algebraic', () => {
    expect(squareToAlgebraic(0)).toBe('a1')
    expect(squareToAlgebraic(3)).toBe('d1')
    expect(squareToAlgebraic(6)).toBe('g1')
    expect(squareToAlgebraic(7)).toBe('a2')
    expect(squareToAlgebraic(55)).toBe('g8')
  })

  it('should convert algebraic to square', () => {
    expect(algebraicToSquare('a1')).toBe(0)
    expect(algebraicToSquare('d1')).toBe(3)
    expect(algebraicToSquare('g1')).toBe(6)
    expect(algebraicToSquare('a2')).toBe(7)
    expect(algebraicToSquare('g8')).toBe(55)
  })

  it('should be reversible', () => {
    for (let sq = 0; sq < TOTAL_SQUARES; sq++) {
      const algebraic = squareToAlgebraic(sq)
      expect(algebraicToSquare(algebraic)).toBe(sq)
    }
  })
})

describe('Bitboard operations', () => {
  it('should detect piece on square', () => {
    // Bitboard with piece on square 3 (d1)
    const bitboard = 8 // 2^3 = 8
    expect(hasPiece(bitboard, 3)).toBe(true)
    expect(hasPiece(bitboard, 0)).toBe(false)
    expect(hasPiece(bitboard, 4)).toBe(false)
  })

  it('should detect multiple pieces', () => {
    // Pieces on squares 0, 1, 2 (a1, b1, c1)
    const bitboard = 7 // 1 + 2 + 4 = 7
    expect(hasPiece(bitboard, 0)).toBe(true)
    expect(hasPiece(bitboard, 1)).toBe(true)
    expect(hasPiece(bitboard, 2)).toBe(true)
    expect(hasPiece(bitboard, 3)).toBe(false)
  })

  it('should get all piece positions', () => {
    const bitboard = 7 // Pieces on 0, 1, 2
    const positions = getPiecePositions(bitboard)
    expect(positions).toEqual([0, 1, 2])
  })

  it('should handle empty bitboard', () => {
    expect(getPiecePositions(0)).toEqual([])
  })

  it('should handle initial P1 pieces', () => {
    // Initial P1 setup: pieces on b1, c1, e1, f1 (squares 1, 2, 4, 5)
    const p1_pieces = 62 // binary: 111110
    const positions = getPiecePositions(p1_pieces)
    expect(positions).toContain(1)
    expect(positions).toContain(2)
    expect(positions).toContain(4)
    expect(positions).toContain(5)
  })
})
