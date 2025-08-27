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
    consensus_reached(false), // Initialize consensus_reached
    current_trace_confidences(), // Initialize current_trace_confidences
    warmup_c_least_scores() { // Initialize warmup_c_least_scores
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
    state->current_trace_confidences.clear();
    state->warmup_c_least_scores.clear(); // Clear warmup scores on reset
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
    state->current_trace_confidences.push_back(token_confidence); // Store confidence for C_tail and C_bottom-N
    
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
    
    // Handle warmup mode transitions
    if (state->params.warmup_enabled && state->warmup_mode == deepconf_state::DEEPCONF_WARMUP_MODE_COLLECT) {
        // Check if we've collected enough traces
        if ((int)state->warmup_c_least_scores.size() >= state->params.warmup_traces) {
            LOG_INF("DeepConf: Collected enough warmup traces (%zu), setting adaptive threshold\n",
                    state->warmup_c_least_scores.size());
            // Calculate and set adaptive threshold
            deepconf_set_adaptive_threshold(state, state->params.warmup_percentile);
            // Transition to APPLY mode
            deepconf_set_warmup_mode(state, deepconf_state::DEEPCONF_WARMUP_MODE_APPLY);
            // Reset state for real generation
            deepconf_reset(state);
        }
    }
    
    return should_continue;
}

// Function to be called when a trace is finished to collect C_least score during warmup
void deepconf_end_trace(deepconf_state * state) {
    if (!state || !state->params.warmup_enabled) {
        return;
    }
    
    // If we're in collection mode, collect the C_least score for this trace
    if (state->warmup_mode == deepconf_state::DEEPCONF_WARMUP_MODE_COLLECT) {
        // Store the minimum group confidence (C_least) for this trace
        state->warmup_c_least_scores.push_back(state->min_group_confidence);
        LOG_INF("DeepConf: Collected C_least score %.6f for trace %zu\n",
                state->min_group_confidence, state->warmup_c_least_scores.size());
        
        // Check if we've collected enough traces
        if ((int)state->warmup_c_least_scores.size() >= state->params.warmup_traces) {
            LOG_INF("DeepConf: Collected enough warmup traces (%zu), setting adaptive threshold\n",
                    state->warmup_c_least_scores.size());
            
            // Temporarily replace the confidence window with our collected scores for percentile calculation
            ring_buffer<float> * original_window = state->confidence_window;
            state->confidence_window = new ring_buffer<float>(state->warmup_c_least_scores.size());
            
            // Fill the window with our collected scores
            for (float score : state->warmup_c_least_scores) {
                state->confidence_window->push_back(score);
            }
            
            // Calculate and set adaptive threshold
            deepconf_set_adaptive_threshold(state, state->params.warmup_percentile);
            
            // Restore original window
            delete state->confidence_window;
            state->confidence_window = original_window;
            
            // Transition to APPLY mode
            deepconf_set_warmup_mode(state, deepconf_state::DEEPCONF_WARMUP_MODE_APPLY);
        }
    }
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
    if (!state) {
        return;
    }
    
    // Use the warmup_c_least_scores if we're in warmup mode, otherwise use the confidence window
    std::vector<float> scores;
    if (state->params.warmup_enabled &&
        (state->warmup_mode == deepconf_state::DEEPCONF_WARMUP_MODE_COLLECT ||
         state->warmup_mode == deepconf_state::DEEPCONF_WARMUP_MODE_APPLY)) {
        // Use collected C_least scores during warmup
        scores = state->warmup_c_least_scores;
    } else {
        // Convert ring_buffer to vector for percentile calculation
        scores = state->confidence_window->to_vector();
    }
    
    // Check if we have any scores
    if (scores.empty()) {
        LOG_WRN("No scores available for adaptive threshold calculation\n");
        return;
    }
    
    // Calculate the percentile-based threshold
    float adaptive_threshold = deepconf_calculate_percentile(scores, percentile);
    
    // Set the calculated threshold
    deepconf_set_stopping_threshold(state, adaptive_threshold);
    
    LOG_INF("Set adaptive threshold to %.6f based on %d percentile of %zu scores\n",
            adaptive_threshold, percentile, scores.size());
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
    params.tail_size = 5; // Default for C_tail
    params.bottom_n = 10; // Default for C_bottom-N
    return params;
}

float deepconf_calculate_c_tail(const deepconf_state * state) {
    if (!state || state->current_trace_confidences.empty() || state->params.tail_size == 0) {
        return 0.0f;
    }

    size_t start_index = 0;
    if (state->current_trace_confidences.size() > state->params.tail_size) {
        start_index = state->current_trace_confidences.size() - state->params.tail_size;
    }

    double sum_conf = 0.0;
    for (size_t i = start_index; i < state->current_trace_confidences.size(); ++i) {
        sum_conf += state->current_trace_confidences[i];
    }

    return static_cast<float>(sum_conf / (state->current_trace_confidences.size() - start_index));
}

float deepconf_calculate_c_bottom_n(const deepconf_state * state) {
    if (!state || state->current_trace_confidences.empty() || state->params.bottom_n == 0) {
        return 0.0f;
    }
    
    // Sort confidences to find the bottom N
    std::vector<float> sorted_confidences = state->current_trace_confidences;
    std::sort(sorted_confidences.begin(), sorted_confidences.end());
    
    size_t n = std::min(state->params.bottom_n, sorted_confidences.size());
    
    double sum_conf = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_conf += sorted_confidences[i];
    }
    
    return static_cast<float>(sum_conf / n);
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
// Perform confidence-weighted majority voting
// trace_answers: Vector of answer token sequences from different traces
// trace_confidences: Vector of confidence scores for each trace
// Returns: The voted answer token sequence
std::vector<llama_token> deepconf_weighted_majority_voting(
    const std::vector<std::vector<llama_token>> & trace_answers,
    const std::vector<float> & trace_confidences) {
        
    if (trace_answers.empty() || trace_answers.size() != trace_confidences.size()) {
        return {}; // Return empty vector if inputs are inconsistent
    }
        
    // Map to store cumulative confidence weights for each unique answer
    std::map<std::vector<llama_token>, double> answer_weights;
        
    // Accumulate confidence weights for each answer
    for (size_t i = 0; i < trace_answers.size(); ++i) {
        const auto & answer = trace_answers[i];
        double confidence = static_cast<double>(trace_confidences[i]);
        
        // Add confidence to this answer's cumulative weight
        answer_weights[answer] += confidence;
    }
        
    // Find the answer with the highest cumulative confidence weight
    std::vector<llama_token> best_answer;
    double max_weight = -1.0;
    
    for (const auto & pair : answer_weights) {
        if (pair.second > max_weight) {
            max_weight = pair.second;
            best_answer = pair.first;
        }
    }
    
    return best_answer;
}
std::string deepconf_params_to_string(const deepconf_params & params) {
    std::ostringstream oss;
    oss << "deepconf_enabled = " << (params.enabled ? "true" : "false")
        << ", deepconf_window_size = " << params.window_size
        << ", deepconf_threshold = " << params.threshold
        << ", deepconf_top_k = " << params.top_k;
    return oss.str();
}

// Filter traces based on confidence percentiles
// Returns indices of traces to keep (top gamma_filter_percent% of traces)
std::vector<size_t> deepconf_filter_traces_by_confidence(
    const std::vector<float> & trace_confidences,
    float gamma_filter_percent) {
    
    if (trace_confidences.empty()) {
        return {};
    }
    
    // Clamp gamma_filter_percent to valid range [0.0, 100.0]
    gamma_filter_percent = std::max(0.0f, std::min(100.0f, gamma_filter_percent));
    
    // If gamma_filter_percent is 100.0, keep all traces
    if (gamma_filter_percent >= 100.0f) {
        std::vector<size_t> all_indices(trace_confidences.size());
        for (size_t i = 0; i < trace_confidences.size(); ++i) {
            all_indices[i] = i;
        }
        return all_indices;
    }
    
    // Create vector of (confidence, index) pairs
    std::vector<std::pair<float, size_t>> conf_index_pairs;
    conf_index_pairs.reserve(trace_confidences.size());
    
    for (size_t i = 0; i < trace_confidences.size(); ++i) {
        conf_index_pairs.emplace_back(trace_confidences[i], i);
    }
    
    // Sort by confidence in descending order (highest confidence first)
    std::sort(conf_index_pairs.begin(), conf_index_pairs.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });
    
    // Calculate number of traces to keep
    size_t keep_count = static_cast<size_t>(
        std::ceil(trace_confidences.size() * gamma_filter_percent / 100.0f)
    );
    
    // Ensure we keep at least one trace
    keep_count = std::max(static_cast<size_t>(1), std::min(keep_count, trace_confidences.size()));
    
    // Extract indices of top traces
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(keep_count);
    
    for (size_t i = 0; i < keep_count; ++i) {
        filtered_indices.push_back(conf_index_pairs[i].second);
    }
    
    // Sort indices to maintain original order
    std::sort(filtered_indices.begin(), filtered_indices.end());
    
    return filtered_indices;
}