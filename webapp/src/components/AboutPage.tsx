import { Link } from 'react-router-dom';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-300">
      <div className="max-w-3xl mx-auto px-4 py-8 sm:py-12">
        <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm mb-6 inline-block">
          &larr; Back to KnightBall
        </Link>
        <h1 className="text-2xl sm:text-3xl font-bold text-white mb-8">About KnightBall</h1>

        <div className="space-y-8 text-sm leading-relaxed">
          <section>
            <p>
              KnightBall is a free, open-source digital adaptation of the abstract strategy board
              game <strong className="text-white">Razzle Dazzle</strong> (also published
              as <strong className="text-white">Knight Moves</strong>), designed
              by <strong className="text-white">Donald P. Green</strong>.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-3">The Creator</h2>
            <p>
              Don Green is a professor of political science &mdash; currently the J.W. Burgess
              Professor at Columbia University, and previously at Yale, where he directed the
              Institution for Social and Policy Studies from 1996 to 2011. Outside the lecture
              hall, Green is an avid woodworker and game designer. His best-known
              creation, <strong className="text-white">OCTI</strong>, was published in 1999 by
              The Great American Trading Company and was named Best Abstract Strategy Game of
              the Year by <em>Games</em> magazine. He has gone on to invent a family of abstract
              strategy games including OCTI-for-Kids, Jumpin' Java, Mouse Island, Fishpond
              Mancala, Dupe, and &mdash; the inspiration for this site &mdash; Razzle Dazzle.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-3">The Game</h2>
            <p className="mb-3">
              Razzle Dazzle has been described as "basketball meets chess." Each player commands
              a squad of pieces that move like chess knights and a ball that can be passed in
              straight lines between teammates. The goal is simple: get your ball to the
              opposite end of the board before your opponent does the same.
            </p>
            <p>
              The physical game was published by Family Games America &mdash; first
              as <em>Knight Moves</em> in 2005, then repackaged as <em>Razzle
              Dazzle</em>. It shipped with a wooden board, ten wooden blocks (five per
              player), two metal spheres, and a single page of rules. A game typically lasts
              around ten minutes, but beneath the simple rules lies genuine strategic depth:
              positioning your knights to open passing lanes while blocking your opponent's,
              timing forced-pass threats, and choosing when to advance versus when to defend.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-3">How It Works</h2>
            <p className="mb-2">On each turn a player does one of two things:</p>
            <ul className="list-disc list-inside space-y-1 ml-2 text-gray-400">
              <li>
                <strong className="text-gray-300">Move a knight</strong> &mdash; hop one piece
                in an L-shape (two squares plus one), just like a chess knight.
              </li>
              <li>
                <strong className="text-gray-300">Pass the ball</strong> &mdash; send the ball
                in a straight line (any of eight directions) to a friendly piece, chaining
                multiple passes in a single turn if available.
              </li>
            </ul>
            <p className="mt-3">
              If your opponent moves a knight next to your ball carrier, you are forced to pass.
              The first player whose ball reaches the far row wins.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-3">From Tabletop to Screen</h2>
            <p>
              KnightBall is an independent project that brings Razzle Dazzle to the browser.
              The digital version faithfully implements the original rules on the same 8&times;7
              board and adds features that only a computer can provide: AI opponents trained
              through self-play with Monte Carlo tree search and neural networks, real-time
              online multiplayer, game replays, and an opening explorer. The AI runs
              entirely in your browser &mdash; no server round-trip required.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-white mb-3">Acknowledgments</h2>
            <p>
              KnightBall exists because Don Green designed a
              brilliant little game and shared it with the world. We are grateful for his
              creativity and for the abstract-games community that kept Razzle Dazzle alive.
            </p>
          </section>
        </div>

        <div className="mt-12 pt-6 border-t border-gray-800 text-xs text-gray-600">
          <Link to="/terms" className="hover:text-gray-400">Terms</Link>
          {' \u00B7 '}
          <Link to="/privacy" className="hover:text-gray-400">Privacy</Link>
          {' \u00B7 '}
          <Link to="/" className="hover:text-gray-400">Play KnightBall</Link>
        </div>
      </div>
    </div>
  );
}
