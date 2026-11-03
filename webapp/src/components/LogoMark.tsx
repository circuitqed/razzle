/** The KnightBall mark: a blue knight carrying the ball. */
export default function LogoMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <polygon points="16,3 29,16 16,29 3,16" fill="#3b82f6" stroke="#1e3a8a" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points="16,5.5 26.5,16 5.5,16" fill="#fff" opacity="0.22" />
      <circle cx="16" cy="16" r="5" fill="#fbbf24" stroke="#92400e" strokeWidth="1.2" />
    </svg>
  );
}
