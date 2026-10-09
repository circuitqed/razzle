/*
 * Razzle Dazzle - Fast C engine for MCTS
 *
 * Board: 8 rows x 7 cols = 56 squares, stored in uint64_t bitboards.
 * Square index = row * 7 + col (row 0 = rank 1, col 0 = file a).
 *
 * Move encoding: src * 56 + dst for piece/ball moves, -1 for END_TURN.
 * Action space: 56*56 + 1 = 3137 (index 3136 = END_TURN in policy arrays).
 */

/* ============================================================
 * State
 * ============================================================ */
typedef struct {
    uint64_t pieces[2];       /* Piece bitboards (p1, p2) */
    uint64_t balls[2];        /* Ball bitboards (single bit each) */
    uint64_t touched_mask;    /* Pieces ineligible for passes */
    int32_t  current_player;  /* 0 or 1 */
    int32_t  has_passed;      /* 0 or 1 */
    int32_t  last_knight_dst; /* -1 or 0-55 */
    int32_t  ply;
} RazzleState;

void     razzle_state_init(RazzleState *s);
void     razzle_state_copy(const RazzleState *src, RazzleState *dst);
void     razzle_state_apply_move(RazzleState *s, int move);
int      razzle_state_is_terminal(const RazzleState *s);
int      razzle_state_get_winner(const RazzleState *s);
float    razzle_state_get_result(const RazzleState *s, int player);
int      razzle_state_get_legal_moves(const RazzleState *s, int *moves_out);
void     razzle_state_to_tensor(const RazzleState *s, float *out);
void     razzle_state_extra_planes(const RazzleState *s, float *out);  /* v2 planes 7-8 */
int      razzle_state_equals(const RazzleState *a, const RazzleState *b);
/* 64-bit hash of the full position (everything but ply): pieces, balls, ineligibility,
 * side to move, last knight destination (forced-pass trigger), mid-pass flag. Repetition
 * only compares turn-start positions (mid-pass flag 0). */
uint64_t razzle_state_hash(const RazzleState *s);

/* ============================================================
 * MCTS Tree
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
    /* Repetition rule (appended: earlier fields keep their offsets for ctypes mirrors).
     * history = hashes of the turn-start positions before the root. With
     * repetition_draw set, a turn-start leaf whose position already occurred twice
     * (history + search path) is a terminal draw. Mid-pass states never count. */
    uint64_t *history;
    int32_t   history_len;
    int32_t   history_cap;
    int32_t   repetition_draw;
    /* Search options (appended; razzle_mcts_set_search_options; all 0 = original search).
     * vloss_q: virtual loss also counts as losses in Q, not only as visits in U, so a
     *   batch of leaves spreads over more children.
     * fpu_mode / fpu_reduction: Q of an unvisited child. 0 = 0 (draw); 1 = the parent's
     *   Q minus fpu_reduction * sqrt(prior mass of its visited children) (Lc0 style). */
    int32_t   vloss_q;
    int32_t   fpu_mode;
    double    fpu_reduction;
} MCTSTree;

MCTSTree *razzle_mcts_create(const RazzleState *root_state, int max_nodes, int max_batch, int max_depth);
/* Set the game history (hashes of positions before the root) and the repetition rule
 * (0 = none, 1 = threefold repetition is a draw). Call again after razzle_mcts_reroot. */
int       razzle_mcts_set_history(MCTSTree *tree, const uint64_t *hashes, int n, int repetition_draw);
/* Search options (see MCTSTree). Defaults: 0, 0, 0.0. Kept across razzle_mcts_reroot. */
void      razzle_mcts_set_search_options(MCTSTree *tree, int vloss_q, int fpu_mode, float fpu_reduction);
void      razzle_mcts_free(MCTSTree *tree);

void razzle_mcts_expand_root(MCTSTree *tree, const float *policy);
void razzle_mcts_add_dirichlet_noise(MCTSTree *tree, float eps,
                                      const float *noise, int noise_len);

int  razzle_mcts_select_leaves(MCTSTree *tree, int batch_size, int vloss,
                                float c_puct, float *tensors_out);

/* For the `count` leaves returned by the last select_leaves: side to move and the
 * v2 extra planes (2*56 floats each). One call instead of per-leaf ctypes work. */
void razzle_mcts_leaf_info(const MCTSTree *tree, int count, int32_t *players_out, float *extras_out);

/* As razzle_mcts_select_leaves, but slot b first descends into root action
 * forced_actions[b] (-9999 = no forcing). For Gumbel root search. */
int  razzle_mcts_select_leaves_forced(MCTSTree *tree, int batch_size, int vloss,
                                      float c_puct, float *tensors_out, const int32_t *forced_actions);

void razzle_mcts_expand_and_backup(MCTSTree *tree, int count,
                                    const float *policies,
                                    const float *values,
                                    int vloss);

void razzle_mcts_get_policy(MCTSTree *tree, float *policy_out, float temperature);

int  razzle_mcts_root_visits(MCTSTree *tree);

int  razzle_mcts_should_stop_early(MCTSTree *tree, int min_sims, float threshold);

int  razzle_mcts_check_immediate_win(MCTSTree *tree);

int  razzle_mcts_get_root_children(MCTSTree *tree, int *actions_out,
                                    int *visits_out, float *values_out,
                                    float *priors_out);

int  razzle_mcts_reroot(MCTSTree *tree, int action);  /* tree reuse */
