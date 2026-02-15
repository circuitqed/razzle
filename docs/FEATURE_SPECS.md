# Feature Specifications

This document contains specifications for planned features.

---

## 1. Resign Button

**Status:** Planned

**Description:**
Add a resign button that allows a player to forfeit the current game.

**Requirements:**
- Add a "Resign" button to the game controls area
- Show confirmation dialog before resigning ("Are you sure you want to resign?")
- On resign:
  - Mark game as finished with opponent as winner
  - Update ELO ratings appropriately (same as a loss)
  - For online games: notify opponent via WebSocket
- Button should be disabled when game is already over
- Consider using a less prominent style (gray, or tucked into a menu) to avoid accidental clicks

**UI Location:**
- Could be in the main button row, or in a dropdown/overflow menu
- On mobile, might want it in an overflow menu to save space

---

## 2. Evaluation Meter (AI Games)

**Status:** Planned

**Description:**
Display a thin vertical meter between the board and move history showing the AI's evaluation of the current position.

**Requirements:**
- Only show when playing against AI
- Thin vertical bar (perhaps 8-12px wide)
- Color gradient: Blue at top (good for player), Red at bottom (good for AI)
- Fill level indicates advantage:
  - 50% = equal position
  - 75% toward blue = player has strong advantage
  - 75% toward red = AI has strong advantage
- Update after each move
- Optionally show numeric value on hover (e.g., "+0.3" or "65% win probability")

**Data Source:**
- Use the value head output from the neural network
- Value is typically in range [-1, 1] where +1 = current player winning
- Need to flip sign based on perspective (player is always blue/player 0)

**Visual Design:**
- Smooth gradient or segmented bar
- Consider adding tick marks at 25%, 50%, 75%
- Animate transitions between values

---

## 3. Mobile Layout - Controls Off-Screen Fix

**Status:** Planned

**Description:**
The row of control buttons (New Game, End Turn, Undo, etc.) is cut off on mobile devices.

**Requirements:**
- Ensure all essential controls are visible without scrolling on mobile
- Options to consider:
  1. Make button row scrollable horizontally
  2. Use smaller buttons on mobile
  3. Move some buttons to an overflow menu (...)
  4. Stack buttons in 2 rows on mobile
  5. Use icon-only buttons on mobile (with tooltips)
- End Turn button is critical and must always be visible
- New Game and Undo are frequently used

**Testing:**
- Test on iPhone SE (smallest common screen: 375px wide)
- Test on typical Android phones (~360-400px wide)
- Test in landscape orientation

**Suggested Approach:**
- Use responsive breakpoint (e.g., `sm:` in Tailwind)
- Primary row: New Game, End Turn, Undo
- Secondary/overflow: Sound, Rules, History, Analysis

---

## 4. Login Button Covered by Title

**Status:** Planned

**Description:**
The user menu / login button in the top-right corner is overlapped by the "Razzle Dazzle" title on some screen sizes.

**Requirements:**
- Ensure the header area properly separates title and user menu
- Options:
  1. Make title and user menu a proper flex header row
  2. Add sufficient padding/margin
  3. Reduce title size on smaller screens
  4. Move title to true center with user menu absolutely positioned

**Current Issue:**
- Title is centered in the page
- User menu is `absolute top-4 right-4`
- On narrow screens they overlap

**Suggested Fix:**
- Create a proper header bar with flexbox: `justify-between`
- Left: (empty or hamburger menu)
- Center: Title
- Right: User menu

---

## 5. Client-Side AI Execution

**Status:** Planned

**Description:**
Currently the AI runs on the server, which creates resource contention when multiple users play simultaneously. Move AI execution to run in the user's browser.

**Requirements:**
- Load neural network model in the browser
- Run MCTS search client-side
- Server only needed for:
  - Game state persistence
  - Online multiplayer
  - User authentication
  - Leaderboards

**Technical Approach:**

### Option A: ONNX.js / ONNX Runtime Web
- Export PyTorch model to ONNX format
- Load and run with onnxruntime-web in browser
- Pros: Good performance, WebGL acceleration
- Cons: Need to convert model, ONNX may not support all ops

### Option B: TensorFlow.js
- Convert PyTorch model to TensorFlow, then to TF.js
- Pros: Mature ecosystem, good WebGL support
- Cons: Two-step conversion, potential accuracy differences

### Option C: WebAssembly + Custom
- Compile MCTS + lightweight inference to WASM
- Pros: Fast, portable
- Cons: Significant development effort

### Recommended: Option A (ONNX Runtime Web)
1. Add model export script: `scripts/export_onnx.py`
2. Create web worker for AI: `webapp/src/workers/ai.worker.ts`
3. Load model on game start (cache in IndexedDB)
4. Run inference in worker to avoid blocking UI
5. Implement MCTS in TypeScript (or compile Python MCTS to WASM)

**Challenges:**
- MCTS implementation in JS/TS (or compile existing Python)
- Model size (need to keep reasonable for download)
- First-load time (model download + initialization)
- Fallback for browsers without WebGL

**Performance Target:**
- 800 simulations should complete in <5 seconds on mid-range devices
- Model should be <20MB for reasonable load time

---

---

## 6. Time Controls

**Status:** Planned

**Description:**
Add optional time limits to games. Each player has a clock that counts down during their turn. If time runs out, that player loses.

### Time Control Formats

**Initial implementation (simpler):**
- Fixed time per player (e.g., 5 minutes, 10 minutes)
- No increment

**Future enhancement:**
- Increment per move (e.g., 5+3 = 5 minutes + 3 seconds per move)
- Delay (grace period before clock starts)

### Preset Options
| Name | Time | Increment | Notes |
|------|------|-----------|-------|
| Bullet | 2 min | 0 | Fast games |
| Blitz | 5 min | 0 | Quick games |
| Rapid | 10 min | 0 | Standard |
| Classical | 20 min | 0 | Longer thinking |
| Custom | User-defined | User-defined | Advanced |

### UI Elements

**Clock Display:**
- Show both clocks prominently (player's clock and opponent's clock)
- Active clock should be highlighted
- Low time warning: yellow at <1 minute, red at <30 seconds
- Optional: tick sound in last 10 seconds

**Position:**
- Desktop: Above/below the board, or in the side panel
- Mobile: Compact display near turn indicator

**Game Setup:**
- Add time control selector when creating a game
- Options: "No limit", presets, or custom
- For AI games: AI uses TimeManager to allocate thinking time

### Engine Integration

**Existing Infrastructure:**
- `TimeManager` class already exists in `razzle/ai/time_manager.py`
- Dynamically allocates simulations based on remaining time
- Estimates ~500 sims/second for time-to-simulation conversion

**Needed Changes:**
1. Pass remaining time to AI move endpoint
2. AI uses TimeManager to determine simulation budget
3. Return time spent with move response
4. Handle timeout (auto-loss if AI exceeds time)

**AI Time Management Strategy:**
- Reserve buffer time (~2 seconds) for safety
- Allocate time proportionally to estimated moves remaining
- Scale by position difficulty (harder positions get more time)
- Early game: spread time evenly
- Endgame: can use remaining time more freely

### Online Multiplayer Considerations

**Server-Side Clock (Recommended):**
- Server tracks authoritative clock state
- Prevents client-side cheating
- Clock starts when opponent's move is received
- Include network latency compensation

**Clock Sync:**
- Send clock state with each move update
- Handle reconnection: restore clock state from server
- Pause clock during disconnect grace period?

**Timeout Handling:**
- Server detects timeout, declares winner
- Notify both players via WebSocket
- Update game status to "finished" with timeout as reason

### Data Model Changes

**Game table additions:**
```sql
time_control_initial INTEGER,  -- Initial time in seconds (NULL = no limit)
time_control_increment INTEGER, -- Increment per move in seconds
time_remaining_p1 INTEGER,  -- P1 remaining time in milliseconds
time_remaining_p2 INTEGER,  -- P2 remaining time in milliseconds
last_clock_update TEXT,  -- Timestamp of last clock update
```

### API Changes

**Create game request:**
```json
{
  "time_control": {
    "initial": 300,
    "increment": 0
  }
}
```

**Game state response additions:**
```json
{
  "time_control": { "initial": 300, "increment": 0 },
  "time_remaining": [285000, 300000],
  "clock_running_for": 0
}
```

### Implementation Phases

**Phase 1: AI Games Only**
- Add time control option to game setup
- Display clocks in UI
- AI uses TimeManager for move timing
- No increment yet

**Phase 2: Online Games**
- Server-side clock management
- Clock sync via WebSocket
- Timeout detection and handling

**Phase 3: Polish**
- Increment support
- Sound effects for low time
- Clock pause during disconnect
- Time odds (asymmetric starting time)

---

## 7. Rematch Button

**Status:** Planned

**Description:**
After a game ends, allow players to quickly start a new game with the same settings. For online games, this sends a rematch request to the opponent.

### Behavior by Game Mode

**AI Games:**
- Show "Rematch" button when game ends
- Starts new game immediately with same settings:
  - Same AI model
  - Same simulation count
  - Same time control (if any)
- Player stays as Blue (or option to swap colors)

**Local 2-Player:**
- Show "Rematch" button when game ends
- Option to swap colors or keep same
- Starts immediately (no confirmation needed)

**Online Games:**
- Show "Rematch" button when game ends
- Sends rematch request to opponent
- Opponent sees: "Opponent wants a rematch" with Accept/Decline buttons
- If accepted: new game starts with colors swapped
- If declined: requester is notified
- Request expires after 60 seconds
- Either player can send the request

### UI Design

**Button Placement:**
- Appears in the game-over state, near "New Game" button
- More prominent than New Game (since it's the likely action)
- Could be: `[Rematch] [New Game]`

**Online Rematch States:**

| State | Your UI | Opponent's UI |
|-------|---------|---------------|
| You request | "Waiting for opponent..." (Cancel) | "Rematch?" (Accept / Decline) |
| They request | "Rematch?" (Accept / Decline) | "Waiting for opponent..." (Cancel) |
| Accepted | Redirect to new game | Redirect to new game |
| Declined | "Opponent declined" | (dismissed) |
| Timeout | "Request expired" | (dismissed) |

**Visual:**
```
┌─────────────────────────────────┐
│         You Win!                │
│                                 │
│   [🔄 Rematch]  [New Game]     │
│                                 │
│   or return to lobby            │
└─────────────────────────────────┘
```

### Color Swapping

**Default behavior:**
- Online: Swap colors each rematch (loser gets first move next game)
- AI: Keep same colors (player is always Blue)
- Local: Prompt or auto-swap

**Optional setting:**
- "Swap colors on rematch" toggle
- "Random colors" option

### API Design

**New endpoints:**

```
POST /games/online/{game_id}/rematch
  Request: { }
  Response: { "status": "pending", "expires_at": "..." }

POST /games/online/{game_id}/rematch/accept
  Request: { }
  Response: { "new_game_id": "...", "your_color": 1 }

POST /games/online/{game_id}/rematch/decline
  Request: { }
  Response: { "status": "declined" }
```

**WebSocket messages:**

```javascript
// Incoming
{ "type": "rematch_requested", "data": { "expires_at": "..." } }
{ "type": "rematch_accepted", "data": { "new_game_id": "...", "your_color": 0 } }
{ "type": "rematch_declined", "data": { } }
{ "type": "rematch_expired", "data": { } }
{ "type": "rematch_cancelled", "data": { } }

// Outgoing (via REST, not WS)
```

### Data Model

**Option A: Separate table**
```sql
CREATE TABLE rematch_requests (
  id TEXT PRIMARY KEY,
  game_id TEXT REFERENCES games(game_id),
  requester_user_id TEXT,
  status TEXT DEFAULT 'pending',  -- pending, accepted, declined, expired, cancelled
  created_at TEXT,
  expires_at TEXT
);
```

**Option B: In-memory only**
- Store pending requests in server memory
- Simpler, but lost on server restart
- Acceptable since requests are short-lived (60 sec)

**Recommended:** Option B for simplicity. Rematch requests are ephemeral.

### Edge Cases

- Opponent disconnected: Show "Opponent is offline" instead of rematch button
- Opponent left the game page: Request times out
- Both request simultaneously: Both accepted, start game
- Game was abandoned (forfeit): Still allow rematch
- Requester leaves before response: Auto-cancel request

### Implementation Phases

**Phase 1: AI and Local games**
- Add rematch button to game-over UI
- Preserve settings and start new game
- Simple, no server changes needed

**Phase 2: Online games**
- Add rematch request/accept/decline endpoints
- WebSocket notifications
- In-memory request tracking
- UI for pending state

---

## 8. Game Chat (Simple)

**Status:** Planned
**Priority:** Nice to have

**Description:**
Simple text chat between players during online games. No complex moderation - this is for friends and family.

### Scope

- Text messages between the two players only
- Available during and after game
- Basic rate limiting (1 msg/sec) to prevent spam
- Max 500 characters per message

### UI

**Desktop:** Small chat panel next to move history, or collapsible
**Mobile:** Tab that switches between moves and chat, badge for unread

### Technical

**WebSocket messages:**
```javascript
// Send
{ "type": "chat", "data": { "message": "nice move!" } }

// Receive
{ "type": "chat", "data": { "from": "opponent", "message": "thanks!", "timestamp": "..." } }
```

**Storage:** Simple table with game_id, user_id, message, timestamp. No complex moderation needed for trusted users.

---

## 9. Undo Request (Online Games)

**Status:** Planned

**Description:**
In online games, allow a player to request an undo. The opponent can accept or decline.

### Behavior

**Request flow:**
1. Player clicks "Request Undo"
2. Opponent sees notification: "Opponent requests to undo the last move" with Accept/Decline
3. If accepted: game state reverts to before the last move
4. If declined: requester is notified, game continues

**UI:**
- Undo button in online games shows "Request Undo" instead of immediate undo
- Request times out after 30 seconds
- Only one pending request at a time

### API Design

**WebSocket messages:**
```javascript
// Request
{ "type": "undo_request", "data": { } }

// Responses
{ "type": "undo_requested", "data": { "expires_at": "..." } }
{ "type": "undo_accepted", "data": { "new_state": {...} } }
{ "type": "undo_declined", "data": { } }
{ "type": "undo_expired", "data": { } }
```

### Implementation Notes
- Similar pattern to rematch request
- In-memory tracking (ephemeral, like rematch)
- Could allow multiple undos per game (with cooldown?)

---

## Low Priority / Future Consideration

The following features are documented but deprioritized. They add complexity that may not be needed for a friends-and-family use case.

### Spectator Mode
- Allow watching live games
- Adds complexity: separate WebSocket handling, privacy controls
- **For now:** Players can share their screen or sit together

### Friend System
- Add friends, see online status, challenge directly
- Adds complexity: presence tracking, friend requests, notifications
- **For now:** Share game codes via text/chat apps - simpler and works fine

### Ranked Matchmaking
- Auto-pair players by skill level
- Adds complexity: queue management, rating ranges, anti-abuse
- **For now:** Not needed - friends just share codes to play each other

---

## Future Ideas (Not Specified)

- Opening explorer
- Puzzle mode
- Achievements
- Tournament system
- Mobile app
