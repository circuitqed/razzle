interface RulesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function RulesModal({ isOpen, onClose }: RulesModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-gray-800 rounded-lg shadow-xl max-w-lg w-full max-h-[80vh] overflow-y-auto p-6">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white text-2xl"
          aria-label="Close"
        >
          &times;
        </button>

        <h2 className="text-2xl font-bold mb-4">Razzle Dazzle Rules</h2>

        <div className="space-y-4 text-gray-300 text-sm">
          <section>
            <h3 className="font-semibold text-white mb-2">Objective</h3>
            <p>
              Move your ball to the opposite end of the board. Blue (Player 1) aims
              for the top row. Red (Player 2) aims for the bottom row.
            </p>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Setup</h3>
            <p>
              Each player starts with 6 pieces and 1 ball on their home row. The
              ball starts on a center piece.
            </p>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Turn Actions</h3>
            <p className="mb-2">On your turn, you must do ONE of the following:</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>
                <strong>Move a piece:</strong> Move one piece like a knight in
                chess (L-shape: 2 squares in one direction, then 1 square
                perpendicular). Pieces cannot land on occupied squares.
              </li>
              <li>
                <strong>Pass the ball:</strong> Pass the ball along a straight
                line (horizontal, vertical, or diagonal) to one of your pieces.
                You can chain multiple passes before ending your turn.
              </li>
            </ul>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Pass Rules</h3>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>The ball cannot pass through other pieces</li>
              <li>
                A piece that has already touched the ball this turn cannot receive
                it again (shown with a red X)
              </li>
              <li>After passing, you can pass again or end your turn</li>
              <li>You cannot move a piece after passing</li>
            </ul>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Forced Pass Rule</h3>
            <p>
              If your opponent moves a piece adjacent to your ball carrier (8
              surrounding squares), you MUST pass the ball if possible. The ball
              will glow when this happens.
            </p>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Winning</h3>
            <p>
              The first player to move their ball onto the opponent's home row
              wins the game.
            </p>
          </section>

          <section>
            <h3 className="font-semibold text-white mb-2">Tips</h3>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>Plan ahead - think about where your pieces need to be</li>
              <li>Use passes to quickly advance the ball</li>
              <li>Block your opponent's passing lanes</li>
              <li>Force your opponent to pass when it benefits you</li>
            </ul>
          </section>
        </div>

        <button
          onClick={onClose}
          className="mt-6 w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
        >
          Got it!
        </button>
      </div>
    </div>
  );
}
