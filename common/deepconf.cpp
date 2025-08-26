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

deepconf_state::deepconf_state(const deepconf_params & p) : params(p), confidence_window(new ring_buffer<float>(p.window_size)), window_sum(0.0), should_stop(false), last_token_confidence(0.0f), last_group_confidence(0.0f), tokens_processed(0) {
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
    state->tokens_processed = 0;
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
    
    // Check if we should stop based on confidence threshold
    // Log detailed information about the check
    LOG_INF("DeepConf Update: window_fill=%zu/%zu, group_conf=%.6f, threshold=%.6f\n",
               count, state->params.window_size, state->last_group_confidence, state->params.threshold);
    bool should_continue = true;
    if (count >= state->params.window_size) {
        // Only check threshold once window is full
        LOG_INF("DeepConf Update: Window full, checking threshold...\n");
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
        return false;
    }
    
    return state->should_stop;
}

deepconf_stats deepconf_get_stats(const deepconf_state * state) {
    deepconf_stats stats = {};
    
    if (state) {
        stats.tokens_processed = state->tokens_processed;
        stats.last_token_confidence = state->last_token_confidence;
        stats.last_group_confidence = state->last_group_confidence;
        stats.window_fill_level = state->confidence_window->size();
        stats.early_stop_triggered = state->should_stop;
        stats.params = state->params;
    }
    
    return stats;
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