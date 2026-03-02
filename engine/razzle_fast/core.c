/*
 * Razzle Dazzle - Fast C engine for MCTS
 *
 * Implements game state, move generation, and MCTS tree search.
 * Designed to be called from Python via cffi.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================
 * Constants (duplicated from header for cffi set_source inline)
 * ============================================================ */
#define ROWS      8
#define COLS      7
#define NUM_SQ   56
#define NUM_ACTIONS 3137
#define END_TURN_ACTION 3136

#define VALID_MASK   0x00FFFFFFFFFFFFFFULL
#define ROW1_MASK    0x7FULL
#define ROW8_MASK    (ROW1_MASK << 49)

#define P1_START_PIECES 0x3EULL
#define P1_START_BALL   0x08ULL
#define P2_START_PIECES (P1_START_PIECES << 49)
#define P2_START_BALL   (P1_START_BALL << 49)

/* ============================================================
 * State struct (matches header)
 * ============================================================ */
typedef struct {
    uint64_t pieces[2];
    uint64_t balls[2];
    uint64_t touched_mask;
    int32_t  current_player;
    int32_t  has_passed;
    int32_t  last_knight_dst;
    int32_t  ply;
} RazzleState;

/* ============================================================
 * MCTS structs (matches header)
 * ============================================================ */
typedef struct {
    RazzleState state;
    int32_t  parent;
    int32_t  first_child;
    int32_t  next_sibling;
    int32_t  parent_action;
    double   prior;
    int32_t  visit_count;
    double   value_sum;
    int32_t  virtual_loss;
    int32_t  is_terminal;
    int32_t  is_expanded;
    int32_t  num_children;
} MCTSNode;

typedef struct {
    MCTSNode *nodes;
    int32_t   capacity;
    int32_t   count;
    int32_t   root;
    int32_t  *path_buf;
    int32_t  *path_lens;
    int32_t  *leaf_indices;
    int32_t   max_batch;
    int32_t   max_depth;
} MCTSTree;

/* ============================================================
 * Precomputed tables
 * ============================================================ */
static uint64_t KNIGHT_ATTACKS[NUM_SQ];
static int      ROTATE_SQ[NUM_SQ];       /* sq -> rotated sq (180 degrees) */
static int      ROTATE_MOVE[NUM_ACTIONS]; /* move -> rotated move */
static int      tables_initialized = 0;

/* Knight move deltas (dr, dc) */
static const int KNIGHT_DR[8] = {-2, -2, -1, -1,  1,  1,  2,  2};
static const int KNIGHT_DC[8] = {-1,  1, -2,  2, -2,  2, -1,  1};

static inline int sq_row(int sq) { return sq / COLS; }
static inline int sq_col(int sq) { return sq % COLS; }
static inline int rowcol_sq(int r, int c) { return r * COLS + c; }
static inline int valid_rc(int r, int c) { return r >= 0 && r < ROWS && c >= 0 && c < COLS; }
static inline uint64_t bit(int sq) { return 1ULL << sq; }

static inline int popcount64(uint64_t x) {
    /* Use builtin if available */
    return __builtin_popcountll(x);
}

static inline int lsb64(uint64_t x) {
    if (x == 0) return -1;
    return __builtin_ctzll(x);
}

static void init_tables(void) {
    if (tables_initialized) return;
    int sq, i;

    /* Knight attacks */
    for (sq = 0; sq < NUM_SQ; sq++) {
        int r = sq_row(sq), c = sq_col(sq);
        uint64_t attacks = 0;
        for (i = 0; i < 8; i++) {
            int nr = r + KNIGHT_DR[i], nc = c + KNIGHT_DC[i];
            if (valid_rc(nr, nc))
                attacks |= bit(rowcol_sq(nr, nc));
        }
        KNIGHT_ATTACKS[sq] = attacks;
    }

    /* Square rotation map (180 degrees): (r,c) -> (7-r, 6-c) */
    for (sq = 0; sq < NUM_SQ; sq++) {
        int r = sq_row(sq), c = sq_col(sq);
        ROTATE_SQ[sq] = rowcol_sq(ROWS - 1 - r, COLS - 1 - c);
    }

    /* Move rotation map */
    for (i = 0; i < NUM_ACTIONS; i++) {
        if (i == END_TURN_ACTION) {
            ROTATE_MOVE[i] = END_TURN_ACTION;
        } else {
            int src = i / NUM_SQ;
            int dst = i % NUM_SQ;
            ROTATE_MOVE[i] = ROTATE_SQ[src] * NUM_SQ + ROTATE_SQ[dst];
        }
    }

    tables_initialized = 1;
}

/* ============================================================
 * State operations
 * ============================================================ */

void razzle_state_init(RazzleState *s) {
    init_tables();
    s->pieces[0] = P1_START_PIECES;
    s->pieces[1] = P2_START_PIECES;
    s->balls[0]  = P1_START_BALL;
    s->balls[1]  = P2_START_BALL;
    s->touched_mask = 0;
    s->current_player = 0;
    s->has_passed = 0;
    s->last_knight_dst = -1;
    s->ply = 0;
}

void razzle_state_copy(const RazzleState *src, RazzleState *dst) {
    memcpy(dst, src, sizeof(RazzleState));
}

void razzle_state_apply_move(RazzleState *s, int move) {
    init_tables();

    if (move == -1) {
        /* END_TURN: switch player, keep touched_mask */
        s->current_player = 1 - s->current_player;
        s->has_passed = 0;
        s->last_knight_dst = -1;
        s->ply++;
        return;
    }

    int src = move / NUM_SQ;
    int dst = move % NUM_SQ;
    uint64_t src_bit = bit(src);
    uint64_t dst_bit = bit(dst);
    int p = s->current_player;

    if (s->balls[p] & src_bit) {
        /* Ball pass: move ball from src to dst */
        s->balls[p] = dst_bit;
        /* Mark both passer and receiver as having touched ball */
        s->touched_mask |= src_bit | dst_bit;
        s->has_passed = 1;
    } else {
        /* Knight move: piece moves from src to dst */
        s->pieces[p] = (s->pieces[p] & ~src_bit) | dst_bit;
        /* Clear ineligibility for the piece that moved */
        s->touched_mask &= ~src_bit;
        /* Knight move ends the turn */
        s->last_knight_dst = dst;
        s->current_player = 1 - p;
        s->has_passed = 0;
        s->ply++;
    }
}

int razzle_state_is_terminal(const RazzleState *s) {
    if (s->balls[0] & ROW8_MASK) return 1;
    if (s->balls[1] & ROW1_MASK) return 1;
    if (s->ply > 200) return 1;
    return 0;
}

int razzle_state_get_winner(const RazzleState *s) {
    if (s->balls[0] & ROW8_MASK) return 0;
    if (s->balls[1] & ROW1_MASK) return 1;
    if (s->ply > 200) return 1 - s->current_player;
    return -1;
}

float razzle_state_get_result(const RazzleState *s, int player) {
    int winner = razzle_state_get_winner(s);
    if (winner < 0) return 0.5f;
    return (winner == player) ? 1.0f : 0.0f;
}

int razzle_state_equals(const RazzleState *a, const RazzleState *b) {
    return (a->pieces[0] == b->pieces[0] &&
            a->pieces[1] == b->pieces[1] &&
            a->balls[0]  == b->balls[0] &&
            a->balls[1]  == b->balls[1] &&
            a->current_player == b->current_player &&
            a->touched_mask == b->touched_mask &&
            a->has_passed == b->has_passed);
}

/* ============================================================
 * Move generation
 * ============================================================ */

/* Check if current player must pass (opponent just moved adjacent to ball) */
static int must_pass(const RazzleState *s) {
    if (s->last_knight_dst < 0) return 0;

    int p = s->current_player;
    uint64_t ball_pos = s->balls[p];
    if (!ball_pos) return 0;

    int ball_sq = lsb64(ball_pos);
    int ball_r = sq_row(ball_sq), ball_c = sq_col(ball_sq);
    int dst_r = sq_row(s->last_knight_dst), dst_c = sq_col(s->last_knight_dst);
    int rd = abs(ball_r - dst_r), cd = abs(ball_c - dst_c);

    return (rd <= 1 && cd <= 1 && (rd > 0 || cd > 0));
}

/* Generate pass moves, append to moves_out. Returns count added. */
static int gen_pass_moves(const RazzleState *s, int *moves_out) {
    int p = s->current_player;
    uint64_t ball_pos = s->balls[p];
    if (!ball_pos) return 0;

    int ball_sq = lsb64(ball_pos);
    int ball_r = sq_row(ball_sq), ball_c = sq_col(ball_sq);

    uint64_t occupied = s->pieces[0] | s->pieces[1] | s->balls[0] | s->balls[1];
    uint64_t valid_receivers = s->pieces[p] & ~s->touched_mask;

    /* 8 ray directions: N, S, E, W, NE, NW, SE, SW */
    static const int DIRS_DR[8] = { 1, -1,  0,  0,  1,  1, -1, -1};
    static const int DIRS_DC[8] = { 0,  0,  1, -1,  1, -1,  1, -1};

    int count = 0;
    int d;
    for (d = 0; d < 8; d++) {
        int r = ball_r + DIRS_DR[d];
        int c = ball_c + DIRS_DC[d];
        while (valid_rc(r, c)) {
            int sq = rowcol_sq(r, c);
            uint64_t sq_bit = bit(sq);
            if (occupied & sq_bit) {
                /* Hit something */
                if (valid_receivers & sq_bit) {
                    /* Our piece, can receive */
                    moves_out[count++] = ball_sq * NUM_SQ + sq;
                }
                break;  /* Can't pass through pieces */
            }
            r += DIRS_DR[d];
            c += DIRS_DC[d];
        }
    }
    return count;
}

/* Generate knight moves, append to moves_out. Returns count added. */
static int gen_knight_moves(const RazzleState *s, int *moves_out) {
    int p = s->current_player;
    uint64_t movable = s->pieces[p] & ~s->balls[p];
    uint64_t occupied = s->pieces[0] | s->pieces[1] | s->balls[0] | s->balls[1];
    uint64_t empty = (~occupied) & VALID_MASK;

    int count = 0;
    uint64_t tmp = movable;
    while (tmp) {
        int src = lsb64(tmp);
        tmp &= tmp - 1;

        uint64_t targets = KNIGHT_ATTACKS[src] & empty;
        uint64_t t2 = targets;
        while (t2) {
            int dst = lsb64(t2);
            t2 &= t2 - 1;
            moves_out[count++] = src * NUM_SQ + dst;
        }
    }
    return count;
}

int razzle_state_get_legal_moves(const RazzleState *s, int *moves_out) {
    init_tables();

    int pass_count = gen_pass_moves(s, moves_out);

    /* Check forced pass (only at start of turn) */
    if (!s->has_passed && must_pass(s) && pass_count > 0) {
        return pass_count;
    }

    if (s->has_passed) {
        /* Already passed: can pass more or end turn */
        moves_out[pass_count] = -1;  /* END_TURN */
        return pass_count + 1;
    }

    /* Start of turn: can pass OR move knight */
    int knight_count = gen_knight_moves(s, moves_out + pass_count);
    return pass_count + knight_count;
}

/* ============================================================
 * Tensor conversion
 * ============================================================ */

void razzle_state_to_tensor(const RazzleState *s, float *out) {
    init_tables();

    /* Output: 7 planes of ROWS x COLS in CHW order = 7 * 8 * 7 = 392 floats */
    memset(out, 0, 7 * ROWS * COLS * sizeof(float));

    int p = s->current_player;
    int opp = 1 - p;

    uint64_t tmp;

    /* Plane 0: current player's pieces */
    tmp = s->pieces[p];
    while (tmp) {
        int sq = lsb64(tmp); tmp &= tmp - 1;
        int r = sq_row(sq), c = sq_col(sq);
        out[0 * ROWS * COLS + r * COLS + c] = 1.0f;
    }

    /* Plane 1: current player's ball */
    tmp = s->balls[p];
    while (tmp) {
        int sq = lsb64(tmp); tmp &= tmp - 1;
        int r = sq_row(sq), c = sq_col(sq);
        out[1 * ROWS * COLS + r * COLS + c] = 1.0f;
    }

    /* Plane 2: opponent's pieces */
    tmp = s->pieces[opp];
    while (tmp) {
        int sq = lsb64(tmp); tmp &= tmp - 1;
        int r = sq_row(sq), c = sq_col(sq);
        out[2 * ROWS * COLS + r * COLS + c] = 1.0f;
    }

    /* Plane 3: opponent's ball */
    tmp = s->balls[opp];
    while (tmp) {
        int sq = lsb64(tmp); tmp &= tmp - 1;
        int r = sq_row(sq), c = sq_col(sq);
        out[3 * ROWS * COLS + r * COLS + c] = 1.0f;
    }

    /* Plane 4: touched mask */
    tmp = s->touched_mask;
    while (tmp) {
        int sq = lsb64(tmp); tmp &= tmp - 1;
        int r = sq_row(sq), c = sq_col(sq);
        out[4 * ROWS * COLS + r * COLS + c] = 1.0f;
    }

    /* Plane 5: always 1 */
    {
        int i;
        float *plane5 = out + 5 * ROWS * COLS;
        for (i = 0; i < ROWS * COLS; i++)
            plane5[i] = 1.0f;
    }

    /* Plane 6: has_passed indicator */
    if (s->has_passed) {
        int i;
        float *plane6 = out + 6 * ROWS * COLS;
        for (i = 0; i < ROWS * COLS; i++)
            plane6[i] = 1.0f;
    }

    /* Rotate 180 for player 1 */
    if (p == 1) {
        /* Flip along both spatial axes: for CHW layout,
         * (c, r, col) -> (c, ROWS-1-r, COLS-1-col) */
        float tmp_buf[7 * ROWS * COLS];
        memcpy(tmp_buf, out, sizeof(tmp_buf));
        int ch, r2, c2;
        for (ch = 0; ch < 7; ch++) {
            for (r2 = 0; r2 < ROWS; r2++) {
                for (c2 = 0; c2 < COLS; c2++) {
                    out[ch * ROWS * COLS + r2 * COLS + c2] =
                        tmp_buf[ch * ROWS * COLS + (ROWS - 1 - r2) * COLS + (COLS - 1 - c2)];
                }
            }
        }
    }
}

/* ============================================================
 * MCTS Tree
 * ============================================================ */

static inline MCTSNode *node_at(MCTSTree *tree, int idx) {
    return &tree->nodes[idx];
}

static int alloc_node(MCTSTree *tree) {
    if (tree->count >= tree->capacity) return -1;
    int idx = tree->count++;
    MCTSNode *n = &tree->nodes[idx];
    memset(n, 0, sizeof(MCTSNode));
    n->parent = -1;
    n->first_child = -1;
    n->next_sibling = -1;
    n->parent_action = -1;
    return idx;
}

MCTSTree *razzle_mcts_create(const RazzleState *root_state, int max_nodes,
                              int max_batch, int max_depth) {
    init_tables();

    MCTSTree *tree = (MCTSTree *)malloc(sizeof(MCTSTree));
    if (!tree) return NULL;

    tree->nodes = (MCTSNode *)calloc(max_nodes, sizeof(MCTSNode));
    tree->capacity = max_nodes;
    tree->count = 0;
    tree->max_batch = max_batch;
    tree->max_depth = max_depth;

    /* Scratch buffers */
    tree->path_buf = (int32_t *)malloc(max_batch * max_depth * sizeof(int32_t));
    tree->path_lens = (int32_t *)malloc(max_batch * sizeof(int32_t));
    tree->leaf_indices = (int32_t *)malloc(max_batch * sizeof(int32_t));

    /* Allocate root */
    tree->root = alloc_node(tree);
    razzle_state_copy(root_state, &tree->nodes[tree->root].state);
    tree->nodes[tree->root].is_terminal = razzle_state_is_terminal(root_state);

    return tree;
}

void razzle_mcts_free(MCTSTree *tree) {
    if (!tree) return;
    free(tree->nodes);
    free(tree->path_buf);
    free(tree->path_lens);
    free(tree->leaf_indices);
    free(tree);
}

/* Expand a node with children based on policy. */
static void expand_node(MCTSTree *tree, int node_idx, const float *policy) {
    MCTSNode *node = node_at(tree, node_idx);
    if (node->is_expanded) return;

    int moves[256];
    int num_moves = razzle_state_get_legal_moves(&node->state, moves);
    if (num_moves == 0) {
        node->is_expanded = 1;
        return;
    }

    /* Sum policy over legal moves for renormalization.
     * Use double to match Python's float64 sum() accumulation. */
    double policy_sum = 0.0;
    int i;
    for (i = 0; i < num_moves; i++) {
        int m = moves[i];
        int idx = (m == -1) ? END_TURN_ACTION : m;
        policy_sum += (double)policy[idx];
    }

    int prev_child = -1;
    for (i = 0; i < num_moves; i++) {
        int m = moves[i];
        int idx = (m == -1) ? END_TURN_ACTION : m;

        int child_idx = alloc_node(tree);
        if (child_idx < 0) break;  /* Out of nodes */

        MCTSNode *child = node_at(tree, child_idx);
        razzle_state_copy(&node->state, &child->state);
        razzle_state_apply_move(&child->state, m);
        child->parent = node_idx;
        child->parent_action = m;
        child->prior = (policy_sum > 0.0) ? ((double)policy[idx] / policy_sum) : (1.0 / num_moves);
        child->is_terminal = razzle_state_is_terminal(&child->state);

        /* Seed terminal children with actual game result */
        if (child->is_terminal) {
            float result = razzle_state_get_result(&child->state, child->state.current_player);
            child->visit_count = 1;
            child->value_sum = 2.0 * result - 1.0;
        }

        /* Link into sibling list */
        if (prev_child < 0) {
            /* Re-read node pointer in case realloc happened (it won't with our
               design, but be safe) */
            node_at(tree, node_idx)->first_child = child_idx;
        } else {
            node_at(tree, prev_child)->next_sibling = child_idx;
        }
        prev_child = child_idx;
    }

    node_at(tree, node_idx)->num_children = (prev_child >= 0) ? num_moves : 0;
    node_at(tree, node_idx)->is_expanded = 1;
}

void razzle_mcts_expand_root(MCTSTree *tree, const float *policy) {
    expand_node(tree, tree->root, policy);
}

void razzle_mcts_add_dirichlet_noise(MCTSTree *tree, float eps,
                                      const float *noise, int noise_len) {
    MCTSNode *root = node_at(tree, tree->root);
    int child_idx = root->first_child;
    int i = 0;
    while (child_idx >= 0 && i < noise_len) {
        MCTSNode *child = node_at(tree, child_idx);
        child->prior = (1.0 - (double)eps) * child->prior + (double)eps * (double)noise[i];
        child_idx = child->next_sibling;
        i++;
    }
}

/* PUCT selection: pick best child.
 * Uses double precision to match Python's float64 arithmetic. */
static int select_child(MCTSTree *tree, int node_idx, float c_puct) {
    MCTSNode *node = node_at(tree, node_idx);
    int parent_visits = node->visit_count + node->virtual_loss;
    int parent_player = node->state.current_player;
    double sqrt_parent = sqrt((double)parent_visits);
    double d_cpuct = (double)c_puct;

    double best_score = -1e30;
    int best_child = -1;

    int child_idx = node->first_child;
    while (child_idx >= 0) {
        MCTSNode *child = node_at(tree, child_idx);
        int adj_visits = child->visit_count + child->virtual_loss;

        double q = 0.0;
        if (child->visit_count > 0) {
            q = child->value_sum / (double)child->visit_count;
        }

        /* Negate Q if player changed (opponent's loss = our gain) */
        if (child->state.current_player != parent_player) {
            q = -q;
        }

        double exploration = d_cpuct * child->prior * sqrt_parent / (1.0 + adj_visits);
        double score = q + exploration;

        if (score > best_score) {
            best_score = score;
            best_child = child_idx;
        }

        child_idx = child->next_sibling;
    }

    return best_child;
}

int razzle_mcts_select_leaves(MCTSTree *tree, int batch_size, int vloss,
                               float c_puct, float *tensors_out) {
    int leaf_count = 0;
    int terminal_count = 0;
    int b;

    /* Track terminal paths for deferred backup (matches Python behavior).
     * Terminal nodes are NOT backed up immediately — this ensures that
     * subsequent leaf selections within the same batch see the same tree
     * state as Python's MCTS. */
    int terminal_batch_indices[256];  /* batch indices of terminal leaves */

    for (b = 0; b < batch_size; b++) {
        int node_idx = tree->root;
        int *path = tree->path_buf + b * tree->max_depth;
        int path_len = 0;
        path[path_len++] = node_idx;

        /* Traverse tree to leaf, applying virtual loss */
        MCTSNode *node = node_at(tree, node_idx);
        while (node->is_expanded && node->first_child >= 0 && !node->is_terminal) {
            node->virtual_loss += vloss;
            node_idx = select_child(tree, node_idx, c_puct);
            if (node_idx < 0) break;
            node = node_at(tree, node_idx);
            if (path_len < tree->max_depth) {
                path[path_len++] = node_idx;
            }
        }

        /* Apply virtual loss to leaf */
        node->virtual_loss += vloss;
        tree->path_lens[b] = path_len;

        if (node->is_terminal) {
            /* Defer terminal backup until after all leaves are selected */
            terminal_batch_indices[terminal_count++] = b;
        } else {
            /* Non-terminal leaf: need NN evaluation */
            tree->leaf_indices[leaf_count] = b;  /* batch index */

            /* Convert state to tensor */
            razzle_state_to_tensor(&node->state,
                tensors_out + leaf_count * 7 * ROWS * COLS);
            leaf_count++;
        }
    }

    /* Now backup all terminal paths (after all selections are done) */
    for (b = 0; b < terminal_count; b++) {
        int batch_idx = terminal_batch_indices[b];
        int *path = tree->path_buf + batch_idx * tree->max_depth;
        int path_len = tree->path_lens[batch_idx];
        MCTSNode *leaf = node_at(tree, path[path_len - 1]);

        int leaf_player = leaf->state.current_player;
        double result = (double)razzle_state_get_result(&leaf->state, leaf_player);
        double value = 2.0 * result - 1.0;

        int k;
        for (k = path_len - 1; k >= 0; k--) {
            MCTSNode *n = node_at(tree, path[k]);
            n->visit_count++;
            n->virtual_loss -= vloss;
            if (n->state.current_player == leaf_player)
                n->value_sum += value;
            else
                n->value_sum -= value;
        }
    }

    return leaf_count;
}

void razzle_mcts_expand_and_backup(MCTSTree *tree, int count,
                                    const float *policies,
                                    const float *values,
                                    int vloss) {
    int i;
    for (i = 0; i < count; i++) {
        int batch_idx = tree->leaf_indices[i];
        int *path = tree->path_buf + batch_idx * tree->max_depth;
        int path_len = tree->path_lens[batch_idx];
        int leaf_node_idx = path[path_len - 1];

        const float *policy = policies + i * NUM_ACTIONS;
        double value = (double)values[i];

        /* Expand the leaf */
        MCTSNode *leaf = node_at(tree, leaf_node_idx);
        if (!leaf->is_expanded) {
            expand_node(tree, leaf_node_idx, policy);
        }

        /* Backup */
        int leaf_player = leaf->state.current_player;
        int k;
        for (k = path_len - 1; k >= 0; k--) {
            MCTSNode *n = node_at(tree, path[k]);
            n->visit_count++;
            n->virtual_loss -= vloss;
            if (n->state.current_player == leaf_player)
                n->value_sum += value;
            else
                n->value_sum -= value;
        }
    }
}

void razzle_mcts_get_policy(MCTSTree *tree, float *policy_out, float temperature) {
    memset(policy_out, 0, NUM_ACTIONS * sizeof(float));

    MCTSNode *root = node_at(tree, tree->root);
    if (root->first_child < 0) return;

    /* Collect visit counts */
    int total_visits = 0;
    int child_idx = root->first_child;
    while (child_idx >= 0) {
        MCTSNode *child = node_at(tree, child_idx);
        total_visits += child->visit_count;
        child_idx = child->next_sibling;
    }

    if (total_visits == 0) return;

    if (temperature <= 0.0f) {
        /* Greedy: all mass on highest visit count */
        int best_action = -1;
        int best_visits = -1;
        child_idx = root->first_child;
        while (child_idx >= 0) {
            MCTSNode *child = node_at(tree, child_idx);
            if (child->visit_count > best_visits) {
                best_visits = child->visit_count;
                int m = child->parent_action;
                best_action = (m == -1) ? END_TURN_ACTION : m;
            }
            child_idx = child->next_sibling;
        }
        if (best_action >= 0)
            policy_out[best_action] = 1.0f;
    } else if (temperature == 1.0f) {
        /* Proportional to visits */
        child_idx = root->first_child;
        while (child_idx >= 0) {
            MCTSNode *child = node_at(tree, child_idx);
            int m = child->parent_action;
            int action_idx = (m == -1) ? END_TURN_ACTION : m;
            policy_out[action_idx] = (float)child->visit_count / (float)total_visits;
            child_idx = child->next_sibling;
        }
    } else {
        /* Temperature-scaled */
        float inv_temp = 1.0f / temperature;
        float sum = 0.0f;
        /* First pass: compute scaled values */
        child_idx = root->first_child;
        while (child_idx >= 0) {
            MCTSNode *child = node_at(tree, child_idx);
            int m = child->parent_action;
            int action_idx = (m == -1) ? END_TURN_ACTION : m;
            float scaled = powf((float)child->visit_count, inv_temp);
            policy_out[action_idx] = scaled;
            sum += scaled;
            child_idx = child->next_sibling;
        }
        /* Normalize */
        if (sum > 0.0f) {
            child_idx = root->first_child;
            while (child_idx >= 0) {
                MCTSNode *child = node_at(tree, child_idx);
                int m = child->parent_action;
                int action_idx = (m == -1) ? END_TURN_ACTION : m;
                policy_out[action_idx] /= sum;
                child_idx = child->next_sibling;
            }
        }
    }
}

int razzle_mcts_root_visits(MCTSTree *tree) {
    return node_at(tree, tree->root)->visit_count;
}

int razzle_mcts_should_stop_early(MCTSTree *tree, int min_sims, float threshold) {
    MCTSNode *root = node_at(tree, tree->root);
    if (root->visit_count < min_sims) return 0;
    if (root->first_child < 0) return 1;

    int total = 0, max_v = 0;
    int child_idx = root->first_child;
    while (child_idx >= 0) {
        MCTSNode *child = node_at(tree, child_idx);
        total += child->visit_count;
        if (child->visit_count > max_v) max_v = child->visit_count;
        child_idx = child->next_sibling;
    }

    if (total == 0) return 0;
    return ((float)max_v / (float)total) >= threshold;
}

int razzle_mcts_check_immediate_win(MCTSTree *tree) {
    MCTSNode *root = node_at(tree, tree->root);
    int root_player = root->state.current_player;

    int child_idx = root->first_child;
    while (child_idx >= 0) {
        MCTSNode *child = node_at(tree, child_idx);
        if (child->is_terminal) {
            float result = razzle_state_get_result(&child->state, root_player);
            if (result == 1.0f) {
                return child->parent_action;
            }
        }
        child_idx = child->next_sibling;
    }
    return -2;  /* No immediate win */
}

int razzle_mcts_get_root_children(MCTSTree *tree, int *actions_out,
                                   int *visits_out, float *values_out,
                                   float *priors_out) {
    MCTSNode *root = node_at(tree, tree->root);
    int root_player = root->state.current_player;
    int count = 0;
    int child_idx = root->first_child;

    while (child_idx >= 0) {
        MCTSNode *child = node_at(tree, child_idx);
        actions_out[count] = child->parent_action;
        visits_out[count] = child->visit_count;
        priors_out[count] = (float)child->prior;

        /* Value from root player's perspective */
        float q = 0.0f;
        if (child->visit_count > 0) {
            q = (float)(child->value_sum / (double)child->visit_count);
            if (child->state.current_player != root_player)
                q = -q;
        }
        values_out[count] = q;

        count++;
        child_idx = child->next_sibling;
    }
    return count;
}
