import { Link } from 'react-router-dom';
import DemoBoard from './DemoBoard';

interface LandingPageProps {
  onPlayNow: () => void;
  onTutorial: () => void;
}

function LogoMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <polygon points="16,3 29,16 16,29 3,16" fill="#3b82f6" stroke="#1e3a8a" strokeWidth="1.5" strokeLinejoin="round" />
      <polygon points="16,5.5 26.5,16 5.5,16" fill="#fff" opacity="0.22" />
      <circle cx="16" cy="16" r="5" fill="#fbbf24" stroke="#92400e" strokeWidth="1.2" />
    </svg>
  );
}

const POINTS = [
  { title: 'Knights move like chess', text: 'One L-shaped hop per turn.' },
  { title: 'The ball moves by passing', text: 'Pass along straight lines, and chain passes in one turn.' },
  { title: 'Reach the far row to win', text: 'Games take about ten minutes.' },
];

export default function LandingPage({ onPlayNow, onTutorial }: LandingPageProps) {
  return (
    <div className="h-[100dvh] overflow-y-auto bg-gray-900 text-white flex flex-col">
      <header className="flex items-center justify-between px-4 sm:px-8 py-4 max-w-6xl w-full mx-auto">
        <div className="flex items-center gap-2">
          <LogoMark className="w-7 h-7" />
          <span className="text-lg font-semibold tracking-tight">KnightBall</span>
        </div>
        <Link to="/about" className="text-sm text-gray-400 hover:text-white transition-colors">About</Link>
      </header>

      <main className="flex-1 w-full max-w-6xl mx-auto px-4 sm:px-8 py-6 sm:py-10 grid gap-10 md:grid-cols-2 md:items-center">
        <section>
          <h1 className="text-4xl sm:text-6xl font-bold tracking-tight mb-4">
            Chess knights.<br />
            <span className="text-blue-400">Playing ball.</span>
          </h1>
          <p className="text-lg text-gray-300 mb-6 max-w-md">
            A quick, deep strategy board game for two. Play the AI, from first steps to grandmaster,
            or challenge a friend online.
          </p>

          <div className="flex flex-wrap items-center gap-3 mb-8">
            <button
              onClick={onPlayNow}
              className="px-7 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-lg font-semibold transition-colors"
            >
              Play now
            </button>
            <button
              onClick={onTutorial}
              className="px-5 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-base font-medium transition-colors"
            >
              Learn in 2 minutes
            </button>
          </div>

          <ul className="space-y-3 max-w-md">
            {POINTS.map((p) => (
              <li key={p.title} className="flex gap-3">
                <svg viewBox="0 0 12 12" className="w-3 h-3 mt-1.5 shrink-0" aria-hidden="true">
                  <polygon points="6,0 12,6 6,12 0,6" fill="#3b82f6" />
                </svg>
                <span>
                  <span className="font-medium text-gray-100">{p.title}.</span>{' '}
                  <span className="text-gray-400">{p.text}</span>
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-6 text-sm text-gray-500">Free, no sign-up needed.</p>
        </section>

        <section className="w-full max-w-sm mx-auto md:max-w-md" aria-label="A sample game">
          <DemoBoard />
          <p className="mt-2 text-center text-xs text-gray-500">A game between two KnightBall AIs</p>
        </section>
      </main>

      <footer className="w-full max-w-6xl mx-auto px-4 sm:px-8 py-6 text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-2 justify-center sm:justify-start">
        <Link to="/about" className="hover:text-gray-300">About</Link>
        <Link to="/support" className="hover:text-gray-300">Support</Link>
        <Link to="/privacy" className="hover:text-gray-300">Privacy</Link>
        <Link to="/terms" className="hover:text-gray-300">Terms</Link>
        <span className="sm:ml-auto">Based on Razzle Dazzle by Donald P. Green</span>
      </footer>
    </div>
  );
}
