/**
 * KnightBall's line icons: 24×24, stroked in currentColor, so they take the
 * button's text colour and hover state. Drawn for this app (no icon font).
 */
import type { SVGProps } from 'react';

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 22, children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

/** My games: a clock face with a rewind arrow. */
export function HistoryIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M3.5 12a8.5 8.5 0 1 0 2.5-6" />
      <path d="M3 3.5v4h4" />
      <path d="M12 7.5V12l3 2" />
    </Icon>
  );
}

/** Report a bug: a small beetle. */
export function BugIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <rect x="7.5" y="8" width="9" height="12" rx="4.5" />
      <path d="M9.5 8a2.5 2.5 0 0 1 5 0" />
      <path d="M12 12v8" />
      <path d="M7.5 12H4M16.5 12H20M7.8 16.5l-3 1.5M16.2 16.5l3 1.5M8.2 9.5 5.5 7.5M15.8 9.5l2.7-2" />
    </Icon>
  );
}

/** How to play (tutorial): a mortarboard. */
export function LearnIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 3 21 8l-9 5-9-5 9-5Z" />
      <path d="M7 10.5V15c0 1.4 2.2 3 5 3s5-1.6 5-3v-4.5" />
      <path d="M21 8v5" />
    </Icon>
  );
}

/** Rules: an open book. */
export function RulesIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M12 6.5C10 5 7 4.5 3.5 5v13c3.5-.5 6.5 0 8.5 1.5 2-1.5 5-2 8.5-1.5V5C17 4.5 14 5 12 6.5Z" />
      <path d="M12 6.5v13" />
    </Icon>
  );
}

/** Sound on: speaker with waves. */
export function SoundOnIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 9.5h3l4.5-4v13L7 14.5H4z" />
      <path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" />
    </Icon>
  );
}

/** Sound off: speaker with a cross. */
export function SoundOffIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 9.5h3l4.5-4v13L7 14.5H4z" />
      <path d="m16 9.5 5 5M21 9.5l-5 5" />
    </Icon>
  );
}

/** Flip board: arrows up and down. */
export function FlipIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 20V4M4.5 7.5 8 4l3.5 3.5" />
      <path d="M16 4v16M12.5 16.5 16 20l3.5-3.5" />
    </Icon>
  );
}
