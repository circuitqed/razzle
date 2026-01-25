# Razzle Dazzle - Task List

## Engine

### Core Game Logic
- [x] Bitboard representation (8x7 = 56 squares)
- [x] Game state management
- [x] Move generation (knight moves + ball passes)
- [x] Win/draw detection
- [x] Move encoding/decoding
- [x] Persistent ineligibility rule (pieces stay ineligible until they move)
- [x] End-turn mechanism for multi-pass turns
- [x] Forced pass rule (when opponent adjacent to ball)

### AI
- [x] MCTS implementation with PUCT
- [x] Neural network architecture (ResNet)
- [x] Batched evaluator for inference
- [x] Dummy evaluator for testing
- [x] Self-play game generation
- [x] Training loop
- [x] Train initial model
- [x] Optimize batched inference for faster self-play

### Server
- [x] FastAPI REST endpoints
- [x] WebSocket support for real-time updates
- [x] Game state management (in-memory)
- [x] Persistent game storage (database)
- [x] Authentication/user accounts

### CLI
- [x] Human vs AI mode
- [x] AI vs AI watch mode
- [x] Ball indicator display
- [x] Game save/load

### Testing
- [x] Bitboard tests
- [x] Move generation tests
- [x] Game state tests
- [x] MCTS tests
- [x] Evaluator tests
- [x] Server API tests
- [x] Integration tests
- [ ] Performance benchmarks

### Infrastructure
- [x] Docker container with hot-reload
- [x] Vast.ai cloud training integration
- [ ] CI/CD pipeline
- [x] Model versioning/registry

---

## Webapp

### UI Components
- [x] Game board component
- [x] Piece rendering
- [x] Move highlighting
- [x] Game status display
- [x] Move history panel

### Game Features
- [x] Single-player vs AI
- [x] Local two-player (hot seat)
- [ ] Online multiplayer
- [x] Difficulty levels (AI strength)
- [x] Undo/redo

### UX
- [x] Mobile responsiveness
- [x] Touch controls
- [x] Animations
- [x] Sound effects
- [x] Dark mode

### Infrastructure
- [x] API client for engine
- [x] WebSocket integration
- [x] State management
- [x] Error handling

---

## Project-Wide

- [x] Monorepo setup
- [x] Docker Compose configuration
- [x] Official rules documentation (engine/docs/RULES.md)
- [ ] API documentation
- [ ] Landing page
- [x] Deployment (production hosting)
