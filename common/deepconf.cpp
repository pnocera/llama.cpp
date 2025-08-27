#include "deepconf.h"
#include "log.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <sstream>

//
// DeepConf state implementation
//

deepconf_state::deepconf_state(const deepconf_params & p) :
    params(p),
    confidence_window(new ring_buffer<float>(p.window_size)),
    window_sum(0.0),
    should_stop(false),
    last_token_confidence(0.0f),
    last_group_confidence(0.0f),
    min_group_confidence(FLT_MAX),
    warmup_mode(DEEPCONF_WARMUP_MODE_NONE),
    tokens_processed(0),
    consensus_reached(false) { // Initialize consensus_reached
}

deepconf_state::~deepconf_state() {
    delete confidence_window;
}

//
// Public API implementation
//

deepconf_state * deepconf_init(const deepconf_params & params) {
    LOG_DBG("DeepConf init called with enabled:%s, window_size:%zu, threshold:%f, top_k:%d\n",
            params.enabled ? "true" : "false", params.window_size, params.threshold, params.top_k);

    // Validate parameters
    deepconf_params validated_params = params;
    bool params_valid = deepconf_validate_params(validated_params);
    if (!params_valid) {
        LOG_WRN("DeepConf parameters were adjusted to valid ranges\n");
    }

    LOG_DBG("DeepConf validated params: enabled:%s, window_size:%zu, threshold:%f, top_k:%d\n",
            validated_params.enabled ? "true" : "false", validated_params.window_size, validated_params.threshold, validated_params.top_k);

    if (!validated_params.enabled) {
        LOG_DBG("DeepConf returning nullptr because enabled is false\n");
        return nullptr; // No state needed when disabled
    }

    // Additional validation to ensure parameters are reasonable
    if (validated_params.window_size == 0) {
        LOG_ERR("DeepConf window_size is zero, disabling DeepConf\n");
        return nullptr;
    }
    
    if (validated_params.top_k <= 0) {
        LOG_ERR("DeepConf top_k is invalid (%d), disabling DeepConf\n", validated_params.top_k);
        return nullptr;
    }

    try {
        LOG_DBG("Creating new deepconf_state with enabled:%s\n", validated_params.enabled ? "true" : "false");
        return new deepconf_state(validated_params);
    } catch (const std::exception & e) {
        LOG_ERR("Failed to initialize DeepConf state: %s\n", e.what());
        return nullptr;
    } catch (...) {
        LOG_ERR("Failed to initialize DeepConf state: unknown exception\n");
        return nullptr;
    }
}

void deepconf_free(deepconf_state * state) {
    delete state;
}

void deepconf_reset(deepconf_state * state) {
    if (!state) return;
    
    state->confidence_window->clear();
    state->window_sum = 0.0;
    state->should_stop = false;
    state->last_token_confidence = 0.0f;
    state->last_group_confidence = 0.0f;
    state->min_group_confidence = FLT_MAX;                          // Reset C_least
    state->warmup_mode = deepconf_state::DEEPCONF_WARMUP_MODE_NONE; // Reset warmup mode
    state->tokens_processed = 0;
    state->answer_votes.clear();
    state->consensus_reached    = false;
    state->current_trace_tokens.clear();
}

void deepconf_add_token_to_current_trace(deepconf_state * state, llama_token token) {
    if (!state) {
        return;
    }
    state->current_trace_tokens.push_back(token);
}

void deepconf_clear_current_trace_tokens(deepconf_state * state) {
    if (!state) {
        return;
    }
    state->current_trace_tokens.clear();
}

const std::vector<llama_token> & deepconf_get_current_trace_tokens(const deepconf_state * state) {
    static const std::vector<llama_token> empty; // Static empty vector to return if state is null
    if (!state) {
        return empty;
    }
    return state->current_trace_tokens;
}

void deepconf_add_answer_vote(deepconf_state * state, const std::vector<llama_token> & answer) {
    if (!state || !state->params.ensemble_enabled) {
        return;
    }
    state->answer_votes[answer]++;
}

void deepconf_calculate_consensus(deepconf_state * state) {
    if (!state || !state->params.ensemble_enabled) {
        return;
    }

    if (state->answer_votes.empty()) {
        state->consensus_reached = false;
        return;
    }

    int total_votes = 0;
    int max_votes   = 0;

    for (const auto & pair : state->answer_votes) {
        total_votes += pair.second;
        if (pair.second > max_votes) {
            max_votes = pair.second;
        }
    }

    if (total_votes == 0) {
        state->consensus_reached = false;
        return;
    }

    float beta = (float)max_votes / total_votes;
    state->consensus_reached = (beta >= state->params.consensus_threshold);
}

bool deepconf_check_consensus(const deepconf_state * state) {
    if (!state || !state->params.ensemble_enabled) {
        return false;
    }
    return state->consensus_reached;
}

float deepconf_calculate_token_confidence(
    const llama_token_data_array * candidates,
    llama_token winning_token,
    int top_k
) {
    // "top_k" here is the number of runner-up tokens (beta) to average over.
    // We compute the top "beta" tokens by probability, excluding the sampled token.
    if (!candidates || candidates->size == 0) {
        return 0.0f;
    }

    // beta = number of runner-up tokens requested
    int beta = top_k > 0 ? top_k : (int) candidates->size;

    // Collect valid candidates (probability in (0, 1]) as (p, index)
    std::vector<std::pair<float, int>> pool;
    pool.reserve(candidates->size);
    for (int i = 0; i < (int) candidates->size; ++i) {
        const float p = candidates->data[i].p;
        if (p > 0.0f && p <= 1.0f) {
            pool.emplace_back(p, i);
        }
    }
    if (pool.empty()) {
        return 0.0f;
    }

    // We only need the top (beta + 1) elements to gather up to "beta" runner-ups after skipping winner.
    // If pool is smaller, clamp accordingly.
    const int need = std::min((int) pool.size(), beta + 1);

    if ((int) pool.size() > need) {
        std::partial_sort(
            pool.begin(), pool.begin() + need, pool.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });
        pool.resize(need);
    } else {
        std::sort(pool.begin(), pool.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    // Sum log-probs of the first "beta" non-winning tokens from the sorted head
    float sum_log_probs = 0.0f;
    int   taken = 0;
    for (int j = 0; j < (int) pool.size() && taken < beta; ++j) {
        const int idx = pool[j].second;
        const llama_token tok = candidates->data[idx].id;
        if (tok == winning_token) {
            continue;
        }
        sum_log_probs += logf(pool[j].first);
        taken++;
    }
    
    

    if (taken == 0) {
        return 0.0f;
    }

    return -sum_log_probs / (float) taken;
}
// Calculate percentile of a vector of float values
// percentile: 0-100, where 0 is minimum and 100 is maximum
float deepconf_calculate_percentile(std::vector<float> & values, int percentile) {
    if (values.empty()) {
        return 0.0f;
    }
    
    // Sort the values
    std::sort(values.begin(), values.end());
    
    // Clamp percentile to valid range
    percentile = std::max(0, std::min(100, percentile));
    
    // Calculate the index
    float index = (percentile / 100.0f) * (values.size() - 1);
    int lower_index = static_cast<int>(std::floor(index));
    int upper_index = static_cast<int>(std::ceil(index));
    
    // Handle edge cases
    if (lower_index < 0) lower_index = 0;
    if (upper_index >= (int)values.size()) upper_index = values.size() - 1;
    
    // If indices are the same, return the value
    if (lower_index == upper_index) {
        return values[lower_index];
    }
    
    // Linear interpolation between the two values
    float fraction = index - lower_index;
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index]);
}
bool deepconf_update(deepconf_state * state, float token_confidence) {
    if (!state || !state->params.enabled) {
        return true; // Continue generation
    }

    // Update state
    state->last_token_confidence = token_confidence;
    state->tokens_processed++;
    
    // Maintain O(1) rolling sum and sliding window
    if (state->confidence_window->size() == state->params.window_size) {
        // buffer full: evict oldest and adjust rolling sum
        float oldest = state->confidence_window->front();
        state->window_sum -= static_cast<double>(oldest);
        state->confidence_window->pop_front();
    }
    state->confidence_window->push_back(token_confidence);
    state->window_sum += static_cast<double>(token_confidence);

    // Calculate group confidence (average of window)
    const size_t count = state->confidence_window->size();
    if (count == 0) {
        state->last_group_confidence = 0.0f;
    } else {
        state->last_group_confidence = static_cast<float>(state->window_sum / static_cast<double>(count));
    }

    // Update C_least if window is full
    if (count >= state->params.window_size) {
        state->min_group_confidence = std::min(state->min_group_confidence, state->last_group_confidence);
    }
    
    // Check if we should stop based on confidence threshold
    // Log detailed information about the check
    LOG_INF("DeepConf Update: window_fill=%zu/%zu, group_conf=%.6f, min_group_conf=%.6f, threshold=%.6f\n",
               count, state->params.window_size, state->last_group_confidence, state->min_group_confidence, state->params.threshold);
    bool should_continue = true;
    if (count >= state->params.window_size) {
        // Only check threshold once window is full
        LOG_INF("DeepConf Update: Window full, checking threshold...\n");
        LOG_INF("DeepConf Update: Comparing group_conf (%.6f) with threshold (%.6f)\n", state->last_group_confidence, state->params.threshold);
        if (state->last_group_confidence < state->params.threshold) {
            LOG_INF("DeepConf Update: Triggering early stop (group_conf < threshold)\n");
            state->should_stop = true;
            should_continue = false;
        } else {
            LOG_INF("DeepConf Update: Not stopping (group_conf >= threshold)\n");
        }
    } else {
        LOG_INF("DeepConf Update: Window not full yet, not checking threshold\n");
    }
    
    return should_continue;
}

bool deepconf_process_token(
    deepconf_state * state,
    const llama_token_data_array * candidates,
    llama_token winning_token
) {
    if (!state || !state->params.enabled) {
        return true; // Continue generation
    }

    // Calculate token confidence
    float confidence = deepconf_calculate_token_confidence(candidates, winning_token, state->params.top_k);
    
    // Log detailed confidence information
    LOG_INF("DeepConf: Token %d, Calculated confidence: %.6f\n", winning_token, confidence);
    
    // Update state and get continuation decision
    return deepconf_update(state, confidence);
}

float deepconf_get_group_confidence(const deepconf_state * state) {
    if (!state || !state->params.enabled) {
        return 0.0f;
    }
    
    return state->last_group_confidence;
}

bool deepconf_should_stop(const deepconf_state * state) {
    if (!state || !state->params.enabled) {
        LOG_DBG("DeepConf: should_stop=false (disabled or null state)\n");
        return false;
    }
    
    // If warmup is enabled and we are in collection mode, never stop
    if (state->params.warmup_enabled && state->warmup_mode == deepconf_state::DEEPCONF_WARMUP_MODE_COLLECT) {
        LOG_DBG("DeepConf: should_stop=false (in warmup collection mode)\n");
        return false;
    }
    
    LOG_DBG("DeepConf: should_stop=%s (threshold=%.6f, last_group_conf=%.6f)\n",
               state->should_stop ? "true" : "false",
               state->params.threshold,
               state->last_group_confidence);
    
    return state->should_stop;
}

deepconf_stats deepconf_get_stats(const deepconf_state * state) {
    deepconf_stats stats = {};
    
    if (state) {
        stats.tokens_processed = state->tokens_processed;
        stats.last_token_confidence = state->last_token_confidence;
        stats.last_group_confidence = state->last_group_confidence;
        stats.min_group_confidence = state->min_group_confidence;
        stats.window_fill_level = state->confidence_window->size();
        stats.early_stop_triggered = state->should_stop;
        stats.params = state->params;
    }
    
    return stats;
}

void deepconf_set_warmup_mode(deepconf_state * state, deepconf_state::deepconf_warmup_mode mode) {
    if (state) {
        state->warmup_mode = mode;
    }
}

// Get the minimum group confidence (C_least) observed so far in the current trace
float deepconf_get_min_group_confidence(const deepconf_state * state) {
    if (!state) {
        return 0.0f;
    }
    return state->min_group_confidence;
}

// Set the dynamic stopping threshold for the DeepConf state
void deepconf_set_stopping_threshold(deepconf_state * state, float threshold) {
    if (state) {
        state->params.threshold = threshold;
    }
}

// Set adaptive threshold based on warmup scores and desired aggressiveness
// state: DeepConf state with collected warmup scores
// percentile: Desired percentile (e.g., 90 for aggressive "DeepConf-low", 10 for conservative "DeepConf-high")
void deepconf_set_adaptive_threshold(deepconf_state * state, int percentile) {
    if (!state || !state->params.warmup_enabled) {
        return;
    }
    
    // Convert ring_buffer to vector for percentile calculation
    std::vector<float> warmup_scores = state->confidence_window->to_vector();
    
    // Check if we have enough scores
    if (warmup_scores.size() < (size_t)state->params.warmup_traces) {
        LOG_WRN("Not enough warmup scores collected (%zu < %d), using default threshold\n",
                warmup_scores.size(), state->params.warmup_traces);
        return;
    }
    
    // Calculate the percentile-based threshold
    float adaptive_threshold = deepconf_calculate_percentile(warmup_scores, percentile);
    
    // Set the calculated threshold
    deepconf_set_stopping_threshold(state, adaptive_threshold);
    
    LOG_INF("Set adaptive threshold to %.6f based on %d percentile of %zu warmup scores\n",
            adaptive_threshold, percentile, warmup_scores.size());
}

bool deepconf_validate_params(deepconf_params & params) {
    bool was_valid = true;
    
    // Validate and clamp window_size
    if (params.window_size < 1) {
        params.window_size = 1;
        was_valid = false;
    } else if (params.window_size > 2048) {
        params.window_size = 2048;
        was_valid = false;
    }
    
    // Validate and clamp threshold
    if (params.threshold < 0.1f) {
        params.threshold = 0.1f;
        was_valid = false;
    } else if (params.threshold > 100.0f) {
        params.threshold = 100.0f;
        was_valid = false;
    }
    
    // Validate and clamp top_k
    if (params.top_k < 2) {
        params.top_k = 2;
        was_valid = false;
    } else if (params.top_k > 40) {
        params.top_k = 40;
        was_valid = false;
    }
    
    return was_valid;
}

deepconf_params deepconf_default_params() {
    deepconf_params params;
    params.enabled = false;
    params.window_size = 8;
    params.threshold = 0.8f;
    params.top_k = 4;
    return params;
}

void deepconf_print_state(const deepconf_state * state) {
    if (!state) {
        printf("DeepConf: disabled\n");
        return;
    }
    
    printf("DeepConf State:\n");
    printf("  enabled: %s\n", state->params.enabled ? "true" : "false");
    printf("  window_size: %zu\n", state->params.window_size);
    printf("  threshold: %.3f\n", state->params.threshold);
    printf("  top_k: %d\n", state->params.top_k);
    printf("  tokens_processed: %zu\n", state->tokens_processed);
    printf("  last_token_confidence: %.6f\n", state->last_token_confidence);
    printf("  last_group_confidence: %.6f\n", state->last_group_confidence);
    printf("  min_group_confidence (C_least): %.6f\n", state->min_group_confidence);
    printf("  window_fill_level: %zu/%zu\n",
           state->confidence_window->size(),
           state->params.window_size);
    printf("  should_stop: %s\n", state->should_stop ? "true" : "false");
}

std::string deepconf_params_to_string(const deepconf_params & params) {
    std::ostringstream oss;
    oss << "deepconf_enabled = " << (params.enabled ? "true" : "false")
        << ", deepconf_window_size = " << params.window_size
        << ", deepconf_threshold = " << params.threshold
        << ", deepconf_top_k = " << params.top_k;
    return oss.str();
}