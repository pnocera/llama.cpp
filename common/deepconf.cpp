#include "deepconf.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <sstream>

// Import ring_buffer template from sampling.cpp
// TODO: This should ideally be moved to a common header to avoid duplication
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

//
// DeepConf state implementation
//

deepconf_state::deepconf_state(const deepconf_params & p) : params(p) {
    confidence_window = new ring_buffer<float>(params.window_size);
    should_stop = false;
    last_token_confidence = 0.0f;
    last_group_confidence = 0.0f;
    tokens_processed = 0;
}

deepconf_state::~deepconf_state() {
    delete confidence_window;
}

//
// Public API implementation
//

deepconf_state * deepconf_init(const deepconf_params & params) {
    // Validate parameters
    deepconf_params validated_params = params;
    if (!deepconf_validate_params(validated_params)) {
        LOG_WRN("DeepConf parameters were adjusted to valid ranges\n");
    }

    if (!validated_params.enabled) {
        return nullptr; // No state needed when disabled
    }

    try {
        return new deepconf_state(validated_params);
    } catch (const std::exception & e) {
        LOG_ERR("Failed to initialize DeepConf state: %s\n", e.what());
        return nullptr;
    }
}

void deepconf_free(deepconf_state * state) {
    delete state;
}

void deepconf_reset(deepconf_state * state) {
    if (!state) return;
    
    state->confidence_window->clear();
    state->should_stop = false;
    state->last_token_confidence = 0.0f;
    state->last_group_confidence = 0.0f;
    state->tokens_processed = 0;
}

float deepconf_calculate_token_confidence(
    const llama_token_data_array * candidates, 
    int top_k
) {
    if (!candidates || candidates->size == 0) {
        return 0.0f;
    }

    // Use provided top_k or fallback to all candidates
    int k = (top_k > 0) ? top_k : (int)candidates->size;
    k = std::min(k, (int)candidates->size);

    if (k <= 0) {
        return 0.0f;
    }

    // Calculate sum of log probabilities of top-k tokens
    float sum_log_probs = 0.0f;
    int valid_tokens = 0;

    for (int i = 0; i < k; i++) {
        float prob = candidates->data[i].p;
        if (prob > 0.0f && prob <= 1.0f) { // Validate probability
            sum_log_probs += logf(prob);
            valid_tokens++;
        }
    }

    // Return negative average log probability (higher = less confident)
    if (valid_tokens > 0) {
        return -sum_log_probs / valid_tokens;
    }
    
    return 0.0f;
}

bool deepconf_update(deepconf_state * state, float token_confidence) {
    if (!state || !state->params.enabled) {
        return true; // Continue generation
    }

    // Update state
    state->last_token_confidence = token_confidence;
    state->tokens_processed++;
    
    // Add to sliding window
    state->confidence_window->push_back(token_confidence);
    
    // Calculate group confidence (average of window)
    if (state->confidence_window->empty()) {
        state->last_group_confidence = 0.0f;
    } else {
        float sum = 0.0f;
        size_t count = state->confidence_window->size();
        
        for (size_t i = 0; i < count; i++) {
            sum += state->confidence_window->rat(i);
        }
        
        state->last_group_confidence = sum / count;
    }
    
    // Check if we should stop based on confidence threshold
    bool should_continue = true;
    if (state->confidence_window->size() >= state->params.window_size) {
        // Only check threshold once window is full
        if (state->last_group_confidence < state->params.threshold) {
            state->should_stop = true;
            should_continue = false;
        }
    }
    
    return should_continue;
}

bool deepconf_process_token(
    deepconf_state * state,
    const llama_token_data_array * candidates
) {
    if (!state || !state->params.enabled) {
        return true; // Continue generation
    }

    // Calculate token confidence
    float confidence = deepconf_calculate_token_confidence(candidates, state->params.top_k);
    
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
        stats.window_fill_level = state->confidence_window ? state->confidence_window->size() : 0;
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
    } else if (params.threshold > 2.0f) {
        params.threshold = 2.0f;
        was_valid = false;
    }
    
    // Validate and clamp top_k
    if (params.top_k < 1) {
        params.top_k = 1;
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
           state->confidence_window ? state->confidence_window->size() : 0,
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