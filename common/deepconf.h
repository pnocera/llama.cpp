#pragma once

#include "llama.h"

#include <vector>
#include <cmath>
#include <string>

// Forward declaration of ring_buffer to avoid circular dependency
template<typename T> struct ring_buffer;

//
// DeepConf: Confidence-based early stopping for efficient LLM inference
//
// This module implements the DeepConf algorithm which calculates token confidence
// scores and enables early stopping when confidence falls below a threshold.
//
// Key concepts:
// - Token Confidence: Negative average log-probability of top-k tokens
// - Group Confidence: Moving average of token confidences over a sliding window
// - Early Stopping: Halt generation when group confidence < threshold
//

struct deepconf_params {
    bool   enabled      = false; // Enable DeepConf early stopping
    size_t window_size  = 8;     // Sliding window size for group confidence (1-2048)
    float  threshold    = 0.8f;  // Confidence threshold for early stopping (0.1-2.0)
    int    top_k        = 4;     // Number of top tokens for confidence calculation (1-40)
};

struct deepconf_state {
    deepconf_params params;
    ring_buffer<float> * confidence_window; // Sliding window of confidence scores
    bool should_stop;                       // Flag indicating early stopping should occur
    float last_token_confidence;           // Confidence of the most recent token
    float last_group_confidence;           // Most recent group confidence value
    size_t tokens_processed;               // Total tokens processed (for debugging)

    deepconf_state(const deepconf_params & params);
    ~deepconf_state();
    
    // Disable copy constructor and assignment operator to avoid issues with ring_buffer pointer
    deepconf_state(const deepconf_state &) = delete;
    deepconf_state & operator=(const deepconf_state &) = delete;
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
    const llama_token_data_array * candidates
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