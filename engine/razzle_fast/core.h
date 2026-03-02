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
int      razzle_state_equals(const RazzleState *a, const RazzleState *b);

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
} MCTSTree;

MCTSTree *razzle_mcts_create(const RazzleState *root_state, int max_nodes, int max_batch, int max_depth);
void      razzle_mcts_free(MCTSTree *tree);

void razzle_mcts_expand_root(MCTSTree *tree, const float *policy);
void razzle_mcts_add_dirichlet_noise(MCTSTree *tree, float eps,
                                      const float *noise, int noise_len);

int  razzle_mcts_select_leaves(MCTSTree *tree, int batch_size, int vloss,
                                float c_puct, float *tensors_out);

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
