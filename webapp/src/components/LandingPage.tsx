interface LandingPageProps {
  onPlayNow: () => void;
  onTutorial: () => void;
}

export default function LandingPage({ onPlayNow, onTutorial }: LandingPageProps) {
  return (
    <div className="h-[100dvh] bg-gray-900 flex flex-col items-center justify-center text-white px-4">
      <h1 className="text-5xl sm:text-6xl font-bold mb-3 tracking-tight">KnightBall</h1>
      <p className="text-gray-400 text-lg mb-10">A strategy game of knights and passes</p>
      <button
        onClick={onPlayNow}
        className="px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-lg font-medium transition-colors"
      >
        Play Now
      </button>
      <button
        onClick={onTutorial}
        className="mt-4 text-gray-400 hover:text-white text-sm transition-colors underline underline-offset-2"
      >
        How to Play
      </button>
    </div>
  );
}
