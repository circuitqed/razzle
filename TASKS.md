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
- [ ] Train initial model
- [ ] Optimize batched inference for faster self-play

### Server
- [x] FastAPI REST endpoints
- [x] WebSocket support for real-time updates
- [x] Game state management (in-memory)
- [ ] Persistent game storage (database)
- [ ] Authentication/user accounts

### CLI
- [x] Human vs AI mode
- [x] AI vs AI watch mode
- [x] Ball indicator display
- [ ] Game save/load

### Testing
- [x] Bitboard tests
- [x] Move generation tests
- [x] Game state tests
- [x] MCTS tests
- [x] Evaluator tests
- [x] Server API tests
- [ ] Integration tests
- [ ] Performance benchmarks

### Infrastructure
- [x] Docker container with hot-reload
- [x] Vast.ai cloud training integration
- [ ] CI/CD pipeline
- [ ] Model versioning/registry

---

## Webapp

### UI Components
- [ ] Game board component
- [ ] Piece rendering
- [ ] Move highlighting
- [ ] Game status display
- [ ] Move history panel

### Game Features
- [ ] Single-player vs AI
- [ ] Local two-player (hot seat)
- [ ] Online multiplayer
- [ ] Difficulty levels (AI strength)
- [ ] Undo/redo

### UX
- [ ] Mobile responsiveness
- [ ] Touch controls
- [ ] Animations
- [ ] Sound effects
- [ ] Dark mode

### Infrastructure
- [ ] API client for engine
- [ ] WebSocket integration
- [ ] State management
- [ ] Error handling

---

## Project-Wide

- [x] Monorepo setup
- [x] Docker Compose configuration
- [x] Official rules documentation (engine/docs/RULES.md)
- [ ] API documentation
- [ ] Landing page
- [ ] Deployment (production hosting)

---

## Current Priority

1. **Train initial model** - Verify the training pipeline works end-to-end
2. **Webapp board component** - Get basic game playable in browser
3. **Online multiplayer** - Real-time games between players
