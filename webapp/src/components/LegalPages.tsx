import { Link } from 'react-router-dom';

function LegalLayout({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-300">
      <div className="max-w-3xl mx-auto px-4 py-8 sm:py-12">
        <Link to="/" className="text-blue-400 hover:text-blue-300 text-sm mb-6 inline-block">
          &larr; Back to KnightBall
        </Link>
        <h1 className="text-2xl sm:text-3xl font-bold text-white mb-8">{title}</h1>
        <div className="space-y-6 text-sm leading-relaxed">
          {children}
        </div>
        <div className="mt-12 pt-6 border-t border-gray-800 text-xs text-gray-600">
          <Link to="/about" className="hover:text-gray-400">About</Link>
          {' \u00B7 '}
          <Link to="/terms" className="hover:text-gray-400">Terms</Link>
          {' \u00B7 '}
          <Link to="/privacy" className="hover:text-gray-400">Privacy</Link>
        </div>
      </div>
    </div>
  );
}

export function TermsPage() {
  return (
    <LegalLayout title="Terms of Service">
      <p className="text-gray-400 text-xs">Last updated: March 2026</p>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">1. Acceptance of Terms</h2>
        <p>
          By accessing or using KnightBall ("the Service"), you agree to be bound by these Terms of
          Service. If you do not agree, do not use the Service.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">2. Description of Service</h2>
        <p>
          KnightBall is a free online strategy board game. You can play against AI opponents or other
          players in real-time. The Service is provided as-is for entertainment and educational purposes.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">3. User Accounts</h2>
        <p>
          You may create an account to track your games and ratings. You are responsible for maintaining
          the security of your account credentials. You must not create accounts for abusive purposes
          or impersonate others.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">4. User Conduct</h2>
        <p>You agree not to:</p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-gray-400">
          <li>Use automated tools or bots to play games without disclosure</li>
          <li>Intentionally exploit bugs or vulnerabilities</li>
          <li>Harass, abuse, or threaten other users</li>
          <li>Attempt to disrupt or overload the Service</li>
          <li>Use the Service for any unlawful purpose</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">5. Intellectual Property</h2>
        <p>
          The KnightBall game, including its design, code, and AI models, is the property of the
          KnightBall team. Game records and replays are available for personal, non-commercial use.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">6. Disclaimer of Warranties</h2>
        <p>
          The Service is provided "as is" and "as available" without warranties of any kind, either
          express or implied. We do not guarantee that the Service will be uninterrupted, secure, or
          error-free. AI opponents may behave unpredictably.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">7. Limitation of Liability</h2>
        <p>
          To the fullest extent permitted by law, KnightBall and its operators shall not be liable for
          any indirect, incidental, special, or consequential damages arising from use of the Service.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">8. Termination</h2>
        <p>
          We reserve the right to suspend or terminate accounts that violate these terms. You may
          delete your account at any time.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">9. Changes to Terms</h2>
        <p>
          We may update these terms from time to time. Continued use of the Service after changes
          constitutes acceptance of the updated terms.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">10. Contact</h2>
        <p>
          For questions about these terms, please open an issue on our GitHub repository or
          contact us at the email listed on the project page.
        </p>
      </section>
    </LegalLayout>
  );
}

export function PrivacyPage() {
  return (
    <LegalLayout title="Privacy Policy">
      <p className="text-gray-400 text-xs">Last updated: March 2026</p>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">1. Information We Collect</h2>
        <p>We collect the following types of information:</p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-gray-400">
          <li>
            <strong className="text-gray-300">Account information:</strong> If you register, we store
            your username, display name, and a hashed password. No email is required.
          </li>
          <li>
            <strong className="text-gray-300">Game data:</strong> We record game moves, results,
            timestamps, and ELO ratings for all played games.
          </li>
          <li>
            <strong className="text-gray-300">Technical data:</strong> Standard server logs may include
            IP addresses, browser type, and request timestamps for security and debugging.
          </li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">2. Cookies and Local Storage</h2>
        <p>
          KnightBall uses browser local storage and cookies for the following purposes:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-gray-400">
          <li>
            <strong className="text-gray-300">Authentication:</strong> A JWT token stored in
            localStorage to keep you logged in.
          </li>
          <li>
            <strong className="text-gray-300">Game state:</strong> Your current game is saved in
            localStorage so it survives page refreshes.
          </li>
          <li>
            <strong className="text-gray-300">Preferences:</strong> Sound settings and other UI
            preferences are stored locally.
          </li>
          <li>
            <strong className="text-gray-300">AI models:</strong> Downloaded ONNX model files are
            cached in IndexedDB to avoid re-downloading.
          </li>
        </ul>
        <p className="mt-2">
          We do not use third-party tracking cookies or analytics services.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">3. How We Use Your Information</h2>
        <p>We use collected information to:</p>
        <ul className="list-disc list-inside mt-2 space-y-1 text-gray-400">
          <li>Operate the game and matchmaking system</li>
          <li>Calculate and display player ratings</li>
          <li>Maintain game history and replay functionality</li>
          <li>Improve the AI training pipeline (aggregate game data)</li>
          <li>Debug issues and prevent abuse</li>
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">4. Data Sharing</h2>
        <p>
          We do not sell, rent, or share your personal information with third parties. Game records
          (moves, results) may be publicly visible through the game browser and replay features.
          Aggregated, anonymized game data may be used for AI research.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">5. Data Retention</h2>
        <p>
          Account information and game history are retained as long as your account exists. You may
          request deletion of your account and associated data by contacting us.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">6. Security</h2>
        <p>
          We take reasonable measures to protect your data, including hashing passwords and using
          HTTPS for all connections. However, no system is perfectly secure, and we cannot guarantee
          absolute security.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">7. Children's Privacy</h2>
        <p>
          KnightBall is not directed at children under 13. We do not knowingly collect personal
          information from children under 13.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">8. Changes to This Policy</h2>
        <p>
          We may update this privacy policy from time to time. Changes will be reflected in the
          "Last updated" date above.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-2">9. Contact</h2>
        <p>
          For privacy-related questions or data deletion requests, please open an issue on our GitHub
          repository or contact us at the email listed on the project page.
        </p>
      </section>
    </LegalLayout>
  );
}
