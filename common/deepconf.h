#pragma once

#include "llama.h"

#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>

// TODO: deduplicate with llama-impl.h (this comment remains for future refactoring)
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
// DeepConf: Confidence-based early stopping for efficient LLM inference
//
// This module implements the DeepConf algorithm which calculates token confidence
// scores and enables early stopping when confidence falls below a threshold.
//
// Key concepts:
// - Token Confidence: Negative average log-probability of the top-k non-sampled (runners-up) tokens
// - Group Confidence: Simple moving average (SMA) of token confidences over a sliding window
// - Early Stopping: Halt generation when the window is full and group confidence < threshold (0.1–100.0)
//

struct deepconf_params {
    bool   enabled      = false; // Enable DeepConf early stopping
    // Simple moving average window size for group confidence (1-2048).
    // This is "n" in the spec; early stop is evaluated only when the window is full.
    size_t window_size  = 8;

    // Confidence threshold (s) for early stopping (range: 0.1–100.0).
    // Token confidence is defined as the negative average log-probability of the top-k non-sampled
    // (runners-up) tokens. Group confidence is a simple moving average of the last "window_size" token confidences.
    float  threshold    = 0.8f;

    // Number of runners-up (k) used for token confidence (1-40).
    // This excludes the sampled (winning) token.
    int    top_k        = 4;
};

struct deepconf_state {
    deepconf_params params;
    ring_buffer<float> * confidence_window; // Sliding window of confidence scores
    double window_sum;                      // Rolling sum of values for O(1) SMA
    bool should_stop;                       // Flag indicating early stopping should occur
    float last_token_confidence;            // Confidence of the most recent token
    float last_group_confidence;            // Most recent group confidence value
    size_t tokens_processed;                // Total tokens processed (for debugging)

    deepconf_state(const deepconf_params & params);
    ~deepconf_state();
};

// Initialize DeepConf state with given parameters
// Returns nullptr if parameters are invalid
deepconf_state * deepconf_init(const deepconf_params & params);

// Free DeepConf state
void deepconf_free(deepconf_state * state);

// Reset DeepConf state (clear confidence history but preserve parameters)
void deepconf_reset(deepconf_state * state);

// Calculate token confidence from token probability data
// candidates: Array of token candidates with probabilities
// top_k: Number of top tokens to consider (overrides state->params.top_k if > 0)
// Returns: Token confidence score (higher = less confident)
float deepconf_calculate_token_confidence(
    const llama_token_data_array * candidates,
    llama_token winning_token,
    int top_k = -1
);

// Update DeepConf state with new token confidence
// This function:
// 1. Updates the sliding window with the new confidence
// 2. Calculates the current group confidence
// 3. Determines if early stopping should occur
// 4. Updates internal counters and flags
//
// Returns: true if generation should continue, false if early stopping triggered
bool deepconf_update(deepconf_state * state, float token_confidence);

// Process token sampling and update confidence state
// Convenience function that combines confidence calculation and state update
// Returns: true if generation should continue, false if early stopping triggered
bool deepconf_process_token(
    deepconf_state * state,
    const llama_token_data_array * candidates,
    llama_token winning_token
);

// Get current group confidence (average of sliding window)
// Returns 0.0 if window is empty
float deepconf_get_group_confidence(const deepconf_state * state);

// Check if early stopping should occur based on current state
bool deepconf_should_stop(const deepconf_state * state);

// Get statistics for debugging/monitoring
struct deepconf_stats {
    size_t tokens_processed;
    float  last_token_confidence;
    float  last_group_confidence;
    size_t window_fill_level;       // How many slots in window are filled
    bool   early_stop_triggered;
    deepconf_params params;
};

deepconf_stats deepconf_get_stats(const deepconf_state * state);

// Validate DeepConf parameters and adjust to valid ranges if necessary
// Returns true if parameters were valid, false if adjustments were made
bool deepconf_validate_params(deepconf_params & params);

// Get default DeepConf parameters
deepconf_params deepconf_default_params();

// Print DeepConf configuration and current state (for debugging)
void deepconf_print_state(const deepconf_state * state);

// Print human-readable parameter information
std::string deepconf_params_to_string(const deepconf_params & params);