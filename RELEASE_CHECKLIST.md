# Knightball — Release Checklist

## Rebrand

- [ ] Purchase `knightball.com` (or preferred TLD)
- [ ] Rename project references from "Razzle Dazzle" to "Knightball"
  - [ ] `webapp/index.html` — title, meta description, og tags
  - [ ] `webapp/src/App.tsx` — header title
  - [ ] `webapp/src/components/RulesModal.tsx` — game name in rules text
  - [ ] `docker-compose.yml` — service/container names
  - [ ] `og:url` in `index.html` — update to new domain
- [ ] New favicon / logo for Knightball
- [ ] Social preview image (`og:image`, `twitter:image`) — design a card image for link sharing
- [ ] Update CORS `ALLOWED_ORIGINS` to include new domain
- [ ] DNS + SSL setup for new domain pointing to the server
- [ ] Redirect old domain (if keeping it) to new one

## New Features

### Interactive Tutorial
- [ ] Guided first-game experience that teaches core mechanics step by step
  - [ ] Move a knight
  - [ ] Pass the ball
  - [ ] Chain passes
  - [ ] Forced pass mechanic
  - [ ] Score a goal
- [ ] Trigger for new users (first visit or "How to Play" button)
- [ ] Skippable for returning players
- [ ] Highlight valid moves/targets at each step with prompts

### History of the Game
- [ ] New `/about` or `/history` page crediting Don and the origins of the game
- [ ] Link from footer or header nav
- [ ] Story of how the game was created and evolved

### AI Auto-Matching
- [ ] Design strength-based matchmaking system
  - [ ] Player requests "Play vs AI" → system picks AI difficulty to match their Elo
  - [ ] Define AI strength tiers (e.g., sim counts: 50, 200, 800, 2000+)
  - [ ] Calibrate each tier's approximate Elo by running AI-vs-AI games
  - [ ] Map player Elo ranges to appropriate AI tier
- [ ] Implement auto-match endpoint or client-side logic
  - [ ] New players start with a default mid-tier AI
  - [ ] After N games, use actual Elo to select AI strength
  - [ ] Option to manually override and pick difficulty
- [ ] UI for AI difficulty selection
  - [ ] Default: "Auto (matched to your skill)"
  - [ ] Manual: Easy / Medium / Hard / Expert (or slider)
  - [ ] Show friendly difficulty label instead of "iter_275 - 256 sims"

### Landing Experience
- [ ] First visit should get you into a game as fast as possible
  - [ ] One-click "Play Now" → starts a game vs auto-matched AI immediately
  - [ ] No login required to play your first game
  - [ ] Prompt to create account after first game (to save progress/rating)
- [ ] Brief animated game preview or tagline (not a heavy marketing page)

### Player Profiles
- [ ] Profile page (`/player/:id` or `/profile/:username`)
- [ ] Rating history graph (Elo over time)
- [ ] Win/loss/draw record
- [ ] Recent games list
- [ ] Link to profile from game browser, leaderboard, and online games

## Security & Config (Must Fix)

- [x] Set `JWT_SECRET` to a persistent secret in production env
- [x] Set `ALLOWED_ORIGINS` to the production domain
- [x] Protect training data endpoints with auth
  - [x] `GET /training/games`, `/training/games/all`
  - [x] `GET /training/models/{ver}/download`
- [x] Rate-limit `POST /api/logs` and `POST /api/feedback`
- [x] Guard or remove `/dashboard` route from public access

## Legal

- [x] Terms of service page
- [x] Privacy policy page
- [ ] Cookie notice (auth uses cookies)
- [x] Links to legal pages from footer/registration

## Infrastructure

- [x] Database backups — automated backup strategy for SQLite volume
- [ ] Monitoring / alerting — know when the server goes down
- [ ] Error tracking (Sentry or similar) — catch client-side crashes in production
- [ ] Analytics — basic usage stats (games per day, active users, retention)
- [ ] Load testing — how many concurrent WebSocket games can the server handle?
- [ ] Disable server-side AI endpoints (`POST /games/{id}/ai`)
  - [ ] Remove or gate behind admin flag — too expensive for a single server
  - [x] All AI runs client-side (desktop: ONNX Runtime WebGPU/WASM; iOS: custom WebGL2 GPU shaders)
- [x] PWA support + offline AI play
  - [x] Web app manifest (`manifest.json`) — name, icons, theme color, display: standalone
  - [x] Service worker for offline caching (static assets, WASM files)
  - [x] "Add to Home Screen" — works on iOS Safari and Chrome
  - [x] ONNX model cached in IndexedDB after first download
  - [ ] Verify full offline play works end-to-end

## Anti-Abuse

- [ ] Report player mechanism
- [ ] Block player (hide their games from lobby, prevent challenges)
- [ ] Admin tooling — ability to ban accounts
- [ ] Rate-limit game creation to prevent spam

## Code Cleanup

- [x] Remove debug `console.log` calls (already cleaned up; remaining are legitimate error handlers)
- [x] Add 404 catch-all route (show friendly "page not found")
- [x] Add nginx gzip compression for JS/CSS/SVG
- [x] Lazy-load heavy routes (`TrainingDashboard`, `AnalysisBoard`) with `React.lazy()`
- [x] Consistent `VITE_API_URL` usage across all API modules (removed env var, all use `/api`)

## Auth & Accounts

- [x] Password reset flow ("Forgot password?" on login)
- [ ] Profile editing (change display name, change password)
- [x] Email verification on registration (optional but recommended)

## Player Experience Polish

- [x] Tab title notification when it's your turn ("(Your Turn) KnightBall")
- [ ] Sound for incoming challenge / opponent joined
- [ ] Resign confirmation dialog (prevent misclicks)
- [ ] Client-side replay analysis (position evaluation via ONNX in Web Worker)
- [ ] Draw offer mechanism for online games
- [ ] Spectator mode for live games
- [ ] Accessibility basics (focus trapping in modals, keyboard board navigation)

## SEO / Discoverability

- [x] `sitemap.xml`
- [x] `robots.txt`
- [x] Social preview image (also listed under Rebrand)

## Pre-Deploy Verification

- [ ] `cd webapp && ./node_modules/.bin/tsc --noEmit` — passes clean
- [ ] `cd engine && pytest` — passes
- [ ] `cd webapp && ./node_modules/.bin/vitest run src/engine/__tests__/engine.test.ts` — passes
- [ ] Full user flow test: register → login → play AI → play online → game history → replay → export GIF
- [ ] Mobile test: iPhone Safari + Android Chrome
- [ ] WebSocket reconnection + forfeit countdown verified under real conditions
- [ ] All production env vars set: `JWT_SECRET`, `ALLOWED_ORIGINS`, `TRAINING_API_KEY`
- [ ] Docker build from scratch succeeds: `docker compose up --build`
