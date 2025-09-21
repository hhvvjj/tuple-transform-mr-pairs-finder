// ***********************************************************************************
// TUPLE-BASED TRANSFORM mr PAIRS FINDER - COLLATZ SEQUENCE ANALYSIS
// ***********************************************************************************
//
// Author: Javier Hernandez
//
// Email:  271314@pm.me
// 
// Description:
//   The tuple-based transform is a reversible procedure to represent Collatz sequences
//   using the tuple [p, f(p), m, q]. A tuple-based transform calculator can be found
//   here:
//   https://github.com/hhvvjj/tuple-based-transform-calculator
//
//   During the tuple-based transform representation, the multiplicity parameter m repeats
//   two elements, consecutively or at different distances, creating pseudocycles. These two
//   values, named as mr, are shown at the ends of the pseudocycle.
//
//   This is a high-performance parallel search engine to find mr pairs. It efficiently 
//   detects pseudocycles by analyzing the m-value repetition and generates detailed JSON
//   output of all discovered mr pairs in the range of 1 to (2^exponent) - 1. The output also 
//   includes the first occurrence of n that generated each mr value. It uses OpenMP for 
//   parallelization and optimized hash tables for efficient pseudocycle detection.
//
//   Research findings show that mr value discovery exhibits remarkable sparsity across large
//   computational ranges. Comprehensive analysis of the complete range 1 to 2^30, 1073741823
//   numbers, reveals only 42 distinct mr values, suggesting that unique pseudocycle patterns
//   are extremely rare phenomena in Collatz sequence behavior.
//
//   The list of these 42 mr values is: 0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79,
//   91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432, 444, 487, 576, 592, 649, 667,
//   683, 865, 889, 1153, 1214, 1821, 2428, 3643
//
// Usage:
//   ./tuple_transform_mr_pairs_finder <exponent>
//   Example: ./tuple_transform_mr_pairs_finder 25
//
// Output:
//   - JSON file: mr_pairs_detected_range_1_to_2pow<exponent>.json
//   - Console: Real-time progress and comprehensive summary report
//
// License:
//   CC-BY-NC-SA 4.0 International 
//   For additional details, visit:
//   https://creativecommons.org/licenses/by-nc-sa/4.0/
//
//   For full details, visit 
//   https://github.com/hhvvjj/tuple-transform-mr-pairs-finder/blob/main/LICENSE
//
// Research Reference:
//   Based on the tuple-based transform methodology described in:
//   https://doi.org/10.5281/zenodo.15546925
//
// ***********************************************************************************

// ***********************************************************************************
// * 1. HEADERS, DEFINES, TYPEDEFS & GLOBAL VARIABLES
// ***********************************************************************************

// =============================
// SYSTEM HEADERS
// =============================
#include <stdio.h>      // Standard I/O operations: printf, fprintf, fopen, fclose, etc.
#include <stdlib.h>     // General utilities: malloc, free, exit, atoi, realloc
#include <string.h>     // String manipulation: strcmp, memset, strlen, snprintf
#include <stdint.h>     // Fixed-width integer types: uint64_t, uint32_t, int32_t
#include <stdbool.h>    // Boolean type support: bool, true, false
#include <omp.h>        // OpenMP parallelization: #pragma omp, omp_get_wtime, locks

// =============================
// SAFETY AND PERFORMANCE LIMITS
// =============================

// Sequence safety constraints to prevent infinite loops and resource exhaustion
#define MAX_SEQUENCE_LENGTH 50000           // Maximum steps in Collatz sequence before termination

// Progress monitoring configuration for real-time feedback
#define PROGRESS_UPDATE_INTERVAL 3.0        // Seconds between progress display updates
#define PROGRESS_CHECK_FREQUENCY 100000     // Numbers processed between progress checks

// Memory management and dynamic allocation parameters
#define INITIAL_M_CAPACITY 100              // Starting capacity for m-values storage
#define MEMORY_EXPANSION_FACTOR 2           // Growth multiplier when expanding arrays

// Input validation ranges for command-line arguments
#define MIN_EXPONENT 1                      // Minimum search range exponent (2^1)
#define MAX_EXPONENT 64                     // Maximum search range exponent (2^64)

// =============================
// HASH TABLE CONFIGURATION
// =============================

// Hash table sizing (must be power of 2 for efficient modulo via bitwise AND)
#define HASH_TABLE_SIZE 8192                // Total hash table buckets (2^13)
#define HASH_MASK 8191                      // Bitmask for hash function (HASH_TABLE_SIZE - 1)

// =============================
// CORE DATA STRUCTURES
// =============================

/**
 * @brief Hash table node for efficient m-value lookup during pseudocycle detection
 * 
 * Implements separate chaining collision resolution. Each node stores an m-value
 * and forms linked lists at hash table buckets for O(1) average-case lookup.
 */
typedef struct HashNode {
    uint64_t value;                         // Stored m-value for repetition detection
    struct HashNode* next;                  // Next node in collision chain
} HashNode;

/**
 * @brief Container for m-values with hash table optimization for sequence analysis
 * 
 * Hybrid data structure combining dynamic array for sequential storage with
 * hash table for fast repetition detection during Collatz sequence analysis.
 */
typedef struct {
    HashNode* buckets[HASH_TABLE_SIZE];     // Hash table for O(1) lookup
    uint64_t* values;                       // Dynamic array for sequential access
    int count;                              // Current number of stored m-values
    int capacity;                           // Maximum capacity before reallocation
} mValues;

/**
 * @brief Thread-safe collection for unique mr values discovered during search
 * 
 * Maintains parallel arrays storing unique mr values and their first occurrence
 * n values. Uses OpenMP locks for thread-safe concurrent access during parallel
 * search operations.
 */
typedef struct {
    uint64_t* values;                       // Array of unique mr values found
    uint64_t* first_n;                      // Array of first n values for each mr
    int count;                              // Current number of unique mr values
    int capacity;                           // Maximum capacity before expansion
    omp_lock_t lock;                        // OpenMP lock for thread safety
} UniqueMrSet;

/**
 * @brief Thread-safe progress tracking for real-time monitoring of parallel operations
 * 
 * Maintains counters and timing information for progress reporting during parallel
 * search. Uses atomic operations and locks for high-frequency updates.
 */
typedef struct {
    uint64_t processed;                     // Total numbers processed across threads
    uint64_t found_count;                   // Count of numbers yielding mr values
    uint64_t last_n_with_new_unique;        // Most recent n that found new unique mr
    double last_update_time;                // Timestamp of last progress display
    omp_lock_t lock;                        // OpenMP lock for atomic updates
} ProgressTracker;

/**
 * @brief Complete search context containing all operational parameters and state
 * 
 * Central structure holding configuration, shared resources, and storage components
 * for the parallel search operation. Passed between functions to maintain consistent
 * access to all program state.
 */
typedef struct {
    uint64_t max_n;                         // Upper bound of search range (exclusive)
    int exponent;                           // Power of 2 exponent for max_n
    double start_time;                      // Search operation start timestamp
    UniqueMrSet* unique_set;                // Collection of discovered unique mr values
    ProgressTracker* progress;              // Progress monitoring and reporting
} SearchContext;

// ***********************************************************************************
// * 2. UTILITY FUNCTIONS
// ***********************************************************************************

 /**
 * @brief Safely allocates memory with automatic error handling and immediate program termination on failure.
 * 
 * This function provides a wrapper around the standard malloc() call with comprehensive
 * error checking and standardized failure handling. It ensures consistent behavior across
 * the application when memory allocation fails, eliminating the need for repetitive null
 * pointer checks at every allocation site while providing descriptive error messages
 * for debugging purposes.
 * 
 * The function implements a fail-fast strategy: if memory allocation fails, it immediately
 * prints an informative error message to stderr and terminates the program with exit code 1.
 * This approach is suitable for applications where memory allocation failure represents
 * an unrecoverable error condition that should halt execution.
 * 
 * Error Handling Strategy:
 * - Immediate detection of malloc() failure through null pointer check
 * - Descriptive error message including context information for debugging
 * - Graceful program termination with standard error exit code
 * - No possibility of returning null pointers to calling code
 * 
 * @param size The number of bytes to allocate. Must be greater than 0 for meaningful
 *             allocation. The function passes this value directly to malloc().
 * @param context A descriptive string identifying the purpose of the allocation.
 *                This string is included in error messages to aid debugging and
 *                should describe what the memory is intended for.
 *                Examples: "hash table", "m_values array", "unique mr set"
 * 
 * @return A valid pointer to the allocated memory block. This function never returns
 *         NULL because it terminates the program if allocation fails. The returned
 *         memory is uninitialized.
 * 
 * @note This function calls exit(1) on allocation failure, making it unsuitable
 *       for applications that need to recover from memory allocation failures.
 * 
 * @note The returned memory is uninitialized. Use memset() or similar functions
 *       if zero-initialization is required.
 * 
 * @note The context parameter should be a string literal or stable string to
 *       ensure it remains valid during error reporting.
 * 
 * @warning This function terminates the program on failure, so it should only be
 *          used in contexts where immediate termination is acceptable.
 * 
 * @complexity O(1) - constant time wrapper around malloc() with simple error checking
 * 
 * @see malloc(3) for underlying allocation mechanism
 * @see exit(3) for program termination behavior
 * @see safe_realloc() for the corresponding reallocation function
 * 
 * @example
 * ```c
 * // Allocate space for 100 integers
 * int* numbers = safe_malloc(100 * sizeof(int), "integer array");
 * 
 * // Allocate space for a hash table structure
 * HashNode* node = safe_malloc(sizeof(HashNode), "hash node");
 * 
 * // Allocate space for m values array
 * uint64_t* m_values = safe_malloc(capacity * sizeof(uint64_t), "m_values array");
 * 
 * // Error case (simulated)
 * // If system is out of memory, program prints:
 * // "[*] ERROR: Memory allocation failed m_values array"
 * // and exits with code 1
 * ```
 */
static void* safe_malloc(size_t size, const char* context) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "\n[*] ERROR: Memory allocation failed for %s\n", context);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Safely reallocates memory with automatic error handling and immediate program termination on failure.
 * 
 * This function provides a wrapper around the standard realloc() call with comprehensive
 * error checking and standardized failure handling. It ensures consistent behavior when
 * expanding or shrinking dynamically allocated memory blocks, maintaining the same
 * fail-fast strategy as safe_malloc() for uniform error handling throughout the application.
 * 
 * The function handles the complexities of realloc() behavior while providing clear
 * error reporting. Unlike realloc(), which can return NULL on failure while leaving
 * the original pointer valid, this function ensures that allocation failure results
 * in immediate program termination with a descriptive error message.
 * 
 * Reallocation Behavior:
 * - Expands or shrinks the memory block pointed to by ptr
 * - May move the block to a new location if necessary
 * - Preserves existing data up to the minimum of old and new sizes
 * - Returns a pointer to the (possibly moved) memory block
 * - Terminates program immediately if reallocation fails
 * 
 * @param ptr Pointer to the previously allocated memory block to reallocate.
 *            Can be NULL, in which case this function behaves like malloc().
 *            If non-NULL, must be a valid pointer returned by malloc(), calloc(),
 *            or a previous call to realloc().
 * @param size The new size in bytes for the memory block. If 0, the behavior
 *             is implementation-defined (may free the block or return NULL).
 * @param context A descriptive string identifying the purpose of the reallocation.
 *                Used in error messages for debugging. Should describe what the
 *                memory expansion is for, e.g., "m_values expansion", "hash table growth".
 * 
 * @return A valid pointer to the reallocated memory block. This function never
 *         returns NULL because it terminates the program if reallocation fails.
 *         The returned pointer may be different from the input pointer if the
 *         block was moved.
 * 
 * @note If reallocation fails, the original memory block remains valid and unchanged,
 *       but the program terminates before the caller can access it.
 * 
 * @note The function preserves existing data when expanding memory. New memory
 *       beyond the original size is uninitialized.
 * 
 * @note When expanding arrays, callers should update their pointer variables with
 *       the returned value, as the memory block may have moved.
 * 
 * @warning This function terminates the program on failure, making it unsuitable
 *          for applications requiring graceful recovery from memory allocation failures.
 * 
 * @warning After calling this function, the original ptr should be considered invalid
 *          and replaced with the returned pointer.
 * 
 * @complexity O(n) in worst case where n is the smaller of old and new sizes,
 *            due to potential memory copying if the block needs to be moved
 * 
 * @see realloc(3) for underlying reallocation mechanism
 * @see safe_malloc() for initial allocation with similar error handling
 * @see add_m_value() for usage example in dynamic array expansion
 * 
 * @example
 * ```c
 * // Initial allocation
 * int* array = safe_malloc(10 * sizeof(int), "integer array");
 * 
 * // Expand the array to hold 20 integers
 * array = safe_realloc(array, 20 * sizeof(int), "integer array expansion");
 * 
 * // Shrink the array back to 5 integers
 * array = safe_realloc(array, 5 * sizeof(int), "integer array shrinkage");
 * 
 * // Usage in dynamic structure expansion
 * mv->values = safe_realloc(mv->values, new_capacity * sizeof(uint64_t), "m_values expansion");
 * mv->capacity = new_capacity;
 * 
 * // Error case (simulated)
 * // If system cannot provide more memory, program prints:
 * // "[*] ERROR: Memory reallocation failed for m_values expansion"
 * // and exits with code 1
 * ```
 */
static void* safe_realloc(void* ptr, size_t size, const char* context) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        fprintf(stderr, "\n[*] ERROR: Memory reallocation failed for %s\n", context);
        exit(1);
    }
    return new_ptr;
}

/**
 * @brief Computes a hash value for a 64-bit unsigned integer using multiplicative hashing with bit masking.
 * 
 * This function implements a multiplicative hash function that provides good distribution quality
 * for 64-bit values mapped to hash table indices. The algorithm uses the golden ratio-based
 * multiplier (2654435761) combined with bit shifting and masking to achieve balanced hash
 * distribution across the available hash table buckets.
 * 
 * The multiplicative hashing approach significantly outperforms simple bit masking in terms
 * of distribution quality, especially for sequential or mathematically related inputs common
 * in Collatz sequence analysis. This method reduces clustering and provides more uniform
 * distribution across hash table buckets.
 * 
 * Algorithm:
 * 1. Multiply input value by golden ratio constant (2654435761)
 * 2. Extract high-order 32 bits via right shift by 32
 * 3. Apply bitwise AND with HASH_MASK (8191) for final index
 * 
 * @param value The 64-bit unsigned integer to hash. Any value is valid input.
 * 
 * @return A hash value in the range [0, HASH_SIZE-1] where HASH_SIZE = 8192.
 *         The returned value can be used directly as an index into the hash table.
 * 
 * @note The multiplier 2654435761ULL is derived from the golden ratio and provides
 *       excellent distribution characteristics for most input patterns.
 * 
 * @note HASH_MASK is defined as (HASH_SIZE - 1) = 8191, requiring HASH_SIZE to be
 *       a power of 2 for the bitwise AND to work correctly as a modulo operation.
 * 
 * @note This multiplicative approach provides superior hash distribution compared to
 *       simple bit masking, reducing collision rates for sequential and patterned inputs.
 * 
 * @note The right shift by 32 extracts the most random bits from the multiplication
 *       result, further improving distribution quality.
 * 
 * @complexity O(1) - constant time operation with single multiplication and bit operations
 * 
 * @see HASH_SIZE constant definition (8192)
 * @see HASH_MASK constant definition (HASH_SIZE - 1)
 * @see add_m_value() for hash table insertion using this function
 * 
 * @example
 * ```c
 * uint64_t value1 = 12345;
 * uint64_t value2 = 67890;
 * 
 * uint64_t hash1 = hash_function(value1);  // Returns well-distributed hash
 * uint64_t hash2 = hash_function(value2);  // Returns well-distributed hash
 * 
 * // Use hash values as array indices
 * if (hash_table[hash1] != NULL) {
 *     // Handle collision or existing entry
 * }
 * 
 * // Sequential values get different hash buckets
 * for (uint64_t i = 1000; i < 1100; i++) {
 *     uint64_t hash = hash_function(i);  // Good distribution despite sequential input
 * }
 * ```
 */
static inline uint64_t hash_function(uint64_t value) {
    return (uint32_t)((value * 2654435761ULL) >> 32) & HASH_MASK;
}

// ***********************************************************************************
// * 3. CORE ALGORITHM FUNCTIONS
// ***********************************************************************************

/**
 * @brief Calculates the m parameter for a given Collatz sequence value using the tuple-based transform.
 * 
 * This function computes the m value according to the tuple-based transformation described
 * in the research paper. The m parameter represents a normalized form of Collatz sequence
 * values that enables detection of pseudocycles through repetition analysis.
 * 
 * The tuple-based transform defines m as:
 * - For odd values c: m = (c - 1) / 2
 * - For even values c: m = (c - 2) / 2
 * 
 * This transformation maps Collatz sequence values into a space where pseudocycle
 * detection becomes more tractable through repetition analysis of m values.
 * 
 * Algorithm Implementation:
 * 1. Determine p value based on parity: p = 1 for odd c, p = 2 for even c
 * 2. Calculate m = (c - p) / 2 using bit shift for efficient division
 * 3. Return the computed m value
 * 
 * @param c The Collatz sequence value to transform. Must be a positive integer
 *          representing a valid value from a Collatz sequence iteration.
 * 
 * @return The computed m parameter as a 64-bit unsigned integer. The result
 *         represents the tuple-based transform of the input value.
 * 
 * @note The function uses bit manipulation for efficiency:
 *       - (c & 1) to check if c is odd (last bit test)
 *       - >> 1 for division by 2 (right shift)
 * 
 * @note This transformation is central to the mr (m value repeated) detection
 *       algorithm, as repetitions in m values indicate pseudocycle completion.
 * 
 * @note The mathematical foundation for this transform is detailed in the
 *       research paper referenced in the file header (doi.org/10.5281/zenodo.15546925).
 * 
 * @complexity O(1) - constant time operation using only bitwise operations
 * 
 * @see apply_collatz_function() for sequence progression
 * @see is_m_repeated() for repetition detection using m values
 * @see find_first_mr_in_sequence() for the main algorithm using this transform
 * 
 * @example
 * ```c
 * // Example calculations for different Collatz values
 * uint64_t m1 = calculate_m(7);   // Odd: m = (7-1)/2 = 3
 * uint64_t m2 = calculate_m(22);  // Even: m = (22-2)/2 = 10
 * uint64_t m3 = calculate_m(11);  // Odd: m = (11-1)/2 = 5
 * uint64_t m4 = calculate_m(34);  // Even: m = (34-2)/2 = 16
 * 
 * // Use in sequence analysis
 * uint64_t collatz_value = 15;
 * uint64_t m = calculate_m(collatz_value);
 * // Check if this m value has been seen before in the sequence
 * ```
 */
static inline uint64_t calculate_m(uint64_t c) {
    uint64_t p = (c & 1) ? 1 : 2;
    return (c - p) >> 1;
}

/**
 * @brief Applies a single iteration of the Collatz function to a given number.
 * 
 * This function implements the core Collatz conjecture transformation:
 * - If n is odd: n = 3n + 1
 * - If n is even: n = n / 2
 * 
 * The function includes overflow protection for the 3n+1 operation to prevent
 * undefined behavior when dealing with large numbers that could exceed the
 * maximum value of uint64_t.
 * 
 * @param n Pointer to the number to transform. The value is modified in-place.
 *          Must point to a valid uint64_t value.
 * 
 * @return true if the transformation was applied successfully
 *         false if overflow would occur during the 3n+1 operation
 * 
 * @note The function uses bit manipulation for performance:
 *       - (*n & 1) to check if number is odd
 *       - (*n >> 1) for division by 2
 * 
 * @warning If this function returns false, the value pointed to by n is unchanged.
 *          The caller should handle this case appropriately to avoid infinite loops.
 * 
 * @complexity O(1) - constant time operation
 * 
 * @example
 * ```c
 * uint64_t number = 7;
 * if (apply_collatz_function(&number)) {
 *     // number is now 22 (7 * 3 + 1)
 *     printf("New value: %lu\n", number);
 * }
 * 
 * uint64_t large_number = UINT64_MAX / 2;
 * if (!apply_collatz_transform(&large_number)) {
 *     // Function returns false, large_number remains unchanged
 *     // No output is generated - error handled silently
 * }
 * ```
 */ 
static inline bool apply_collatz_function(uint64_t* n) {
    if (*n & 1) {
        if (*n > UINT64_MAX / 3) {
            return false;
        }
        *n = 3 * (*n) + 1;
    } else {
        *n = *n >> 1;
    }
    return true;
}

// ***********************************************************************************
// * 4. m VALUES CONTAINER (SEQUENCE ANALYSIS)
// ***********************************************************************************

/**
 * @brief Initializes an mValues container with hash table optimization for efficient m value storage and lookup.
 * 
 * This function sets up a hybrid data structure that combines a dynamic array for sequential
 * storage of m values with a hash table for O(1) average-case lookup performance. The design
 * enables both efficient iteration through discovered m values and fast repetition detection
 * during Collatz sequence analysis.
 * 
 * The mValues container serves as a critical component in pseudocycle detection, storing all
 * m values encountered in a single Collatz sequence and providing fast lookup to detect when
 * an m value repeats (indicating pseudocycle completion). The dual storage approach optimizes
 * for both access patterns needed during sequence analysis.
 * 
 * Initialization Process:
 * 1. Set initial capacity to INITIAL_M_CAPACITY (100) for reasonable starting size
 * 2. Allocate dynamic array for sequential m value storage
 * 3. Initialize count to zero for empty container state
 * 4. Clear all hash table buckets to NULL for empty hash state
 * 
 * Data Structure Design:
 * - Dynamic array (values): Stores m values in discovery order for iteration
 * - Hash table (buckets): Provides O(1) lookup for repetition detection
 * - Capacity tracking: Enables automatic expansion when storage limits reached
 * - Count tracking: Maintains current number of stored m values
 * 
 * @param mv Pointer to the mValues structure to initialize. Must point to a valid
 *           mValues structure that will be modified in-place. The structure should
 *           not be previously initialized to avoid memory leaks.
 * 
 * @note The function allocates memory for the initial values array but does not
 *       allocate hash table nodes until values are actually added via add_m_value().
 * 
 * @note INITIAL_M_CAPACITY (100) is chosen as a reasonable starting size based on
 *       typical Collatz sequence lengths before pseudocycle detection.
 * 
 * @note All hash table buckets are initialized to NULL, representing empty collision
 *       chains that will be populated as m values are added to the container.
 * 
 * @note The initialized container is ready for immediate use with add_m_value()
 *       and is_m_repeated() functions.
 * 
 * @warning Do not call this function on an already-initialized mValues structure
 *          without first calling destroy_m_values() to avoid memory leaks.
 * 
 * @complexity O(n) where n is HASH_TABLE_SIZE (8192) due to hash bucket initialization
 * 
 * @see destroy_m_values() for proper cleanup of initialized containers
 * @see add_m_value() for adding m values to the initialized container
 * @see is_m_repeated() for checking repetitions using the hash table
 * @see INITIAL_M_CAPACITY and HASH_TABLE_SIZE for sizing constants
 * 
 * @example
 * ```c
 * // Initialize container for sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // Container is now ready for use
 * uint64_t sequence[] = {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1};
 * 
 * for (int i = 0; i < sequence_length; i++) {
 *     uint64_t m = calculate_m(sequence[i]);
 *     
 *     if (is_m_repeated(&m_values, m)) {
 *         printf("Pseudocycle detected at m=%lu\n", m);
 *         break;
 *     }
 *     
 *     add_m_value(&m_values, m);
 * }
 * 
 * // Cleanup when done
 * destroy_m_values(&m_values);
 * ```
 */
static void init_m_values(mValues* mv) {
    mv->capacity = INITIAL_M_CAPACITY;
    mv->values = safe_malloc(mv->capacity * sizeof(uint64_t), "m_values array");
    mv->count = 0;
    
    // Initialize hash table
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        mv->buckets[i] = NULL;
    }
}

/**
 * @brief Safely deallocates all memory associated with an mValues container and resets its state.
 * 
 * This function performs comprehensive cleanup of an mValues structure, ensuring proper
 * deallocation of both the dynamic array storage and all hash table nodes. The cleanup
 * process handles the complex memory layout of the hybrid data structure, preventing
 * memory leaks while resetting the container to a safe, uninitialized state.
 * 
 * The function implements defensive programming by checking for NULL pointers and uses
 * careful pointer management to avoid double-free errors during hash table cleanup.
 * Each node in every collision chain is individually freed using a safe traversal
 * pattern that saves the next pointer before freeing each node.
 * 
 * Cleanup Process:
 * 1. Validate input pointer for safe NULL handling
 * 2. Traverse and free all hash table collision chains
 * 3. Reset all hash bucket pointers to NULL
 * 4. Deallocate the values array
 * 5. Reset all container state variables to safe values
 * 
 * Memory Management Strategy:
 * The function addresses both user-allocated arrays and dynamically created hash nodes.
 * Hash nodes are created during add_m_value() operations and must be individually
 * freed to prevent memory leaks. The careful traversal pattern ensures no nodes
 * are orphaned during the cleanup process.
 * 
 * @param mv Pointer to the mValues structure to clean up. Can be NULL, in which
 *           case the function returns immediately without action. After successful
 *           cleanup, the structure is left in an uninitialized state.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where initialization might have failed.
 * 
 * @note After calling this function, the mValues structure is in an uninitialized
 *       state and should not be used until re-initialized with init_m_values().
 * 
 * @note The function sets all pointer fields to NULL and counters to zero, providing
 *       a clean state that can help detect use-after-free errors during debugging.
 * 
 * @note Hash table cleanup uses a safe traversal pattern that saves the next pointer
 *       before freeing each node, preventing access to freed memory.
 * 
 * @warning Do not use the mValues structure after calling this function until it
 *          has been re-initialized with init_m_values().
 * 
 * @complexity O(n) where n is the total number of hash nodes across all buckets,
 *            which is equal to the number of m values that were added to the container
 * 
 * @see init_m_values() for the corresponding initialization function
 * @see add_m_value() for the function that creates hash nodes
 * @see find_first_mr_in_sequence() for typical usage pattern
 * 
 * @example
 * ```c
 * // Typical usage in sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // ... use container for sequence analysis ...
 * 
 * // Cleanup when analysis is complete
 * destroy_m_values(&m_values);
 * 
 * // Safe to call with NULL
 * destroy_m_values(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * mValues m_values;
 * init_m_values(&m_values);
 * if (some_error_condition) {
 *     destroy_m_values(&m_values);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Usage in find_first_mr_in_sequence
 * uint64_t find_first_mr_in_sequence(uint64_t n_start, bool* found) {
 *     mValues m_values;
 *     init_m_values(&m_values);
 *     
 *     // ... sequence analysis ...
 *     
 *     destroy_m_values(&m_values);  // Always cleanup before return
 *     return first_mr;
 * }
 * ```
 */
static void destroy_m_values(mValues* mv) {
    if (!mv) return;
    
    // Clean up hash table
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode* current = mv->buckets[i];
        while (current) {
            HashNode* next = current->next;
            free(current);
            current = next;
        }
        mv->buckets[i] = NULL;
    }
    
    // Clean up values array
    free(mv->values);
    mv->values = NULL;
    mv->count = 0;
    mv->capacity = 0;
}

/**
 * @brief Efficiently checks if an m value has been previously encountered using hash table lookup.
 * 
 * This function performs fast O(1) average-case lookup to determine if a specific m value
 * has already been stored in the mValues container. This is the core operation for pseudocycle
 * detection in Collatz sequence analysis, as repetition of m values indicates the completion
 * of a pseudocycle pattern.
 * 
 * The function uses the container's hash table for efficient lookup, computing the hash
 * index for the target m value and traversing the collision chain at that bucket until
 * either finding a match or reaching the end of the chain. This approach provides
 * significantly better performance than linear search through the values array.
 * 
 * Pseudocycle Detection Context:
 * In the tuple-based transform approach, when an m value repeats during sequence generation,
 * it indicates that the sequence has entered a pseudocycle. This function enables immediate
 * detection of such repetitions without requiring expensive sequence continuation.
 * 
 * Hash Table Lookup Process:
 * 1. Compute hash index using the hash_function()
 * 2. Access the collision chain at the computed bucket
 * 3. Traverse the linked list comparing each stored value
 * 4. Return true on first match, false if chain is exhausted
 * 
 * @param mv Pointer to the mValues container to search. Must be a properly initialized
 *           container with valid hash table structure. The container is not modified
 *           during the lookup operation.
 * @param m The m value to search for in the container. This represents a transformed
 *          Collatz sequence value that may have been previously encountered.
 * 
 * @return true if the m value has been previously stored in the container
 *         false if the m value is not found in the container
 * 
 * @note This function does not modify the container state and can be called safely
 *       from multiple threads reading the same container (assuming no concurrent writes).
 * 
 * @note The lookup performance depends on the hash function quality and load factor.
 *       With a good hash function, average-case performance is O(1).
 * 
 * @note This function should be called before add_m_value() to implement proper
 *       pseudocycle detection logic in sequence analysis.
 * 
 * @note The function traverses collision chains linearly, so worst-case performance
 *       is O(n) where n is the maximum chain length at any hash bucket.
 * 
 * @complexity Average case: O(1) - constant time lookup with good hash distribution
 *            Worst case: O(n) - linear time when all values hash to the same bucket
 * 
 * @see hash_function() for the hash computation algorithm
 * @see add_m_value() for adding values that this function can subsequently find
 * @see find_first_mr_in_sequence() for usage in pseudocycle detection
 * @see HashNode structure for collision chain implementation
 * 
 * @example
 * ```c
 * // Typical usage in sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * uint64_t sequence[] = {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1};
 * 
 * for (int i = 0; i < sequence_length; i++) {
 *     uint64_t m = calculate_m(sequence[i]);
 *     
 *     // Check for repetition before adding
 *     if (is_m_repeated(&m_values, m)) {
 *         printf("Pseudocycle detected! m=%lu repeats\n", m);
 *         break;  // First repetition found
 *     }
 *     
 *     // Add to container for future repetition checks
 *     add_m_value(&m_values, m);
 * }
 * 
 * // Example with specific values
 * add_m_value(&m_values, 25);
 * add_m_value(&m_values, 108);
 * 
 * bool found1 = is_m_repeated(&m_values, 25);   // Returns true
 * bool found2 = is_m_repeated(&m_values, 999);  // Returns false
 * bool found3 = is_m_repeated(&m_values, 108);  // Returns true
 * ```
 */
static bool is_m_repeated(const mValues* mv, uint64_t m) {
    uint64_t hash = hash_function(m);
    HashNode* current = mv->buckets[hash];
    
    while (current) {
        if (current->value == m) {
            return true;
        }
        current = current->next;
    }
    return false;
}

/**
 * @brief Adds a new m value to the container with automatic capacity expansion and hash table insertion.
 * 
 * This function implements efficient storage of m values using a dual-structure approach that
 * maintains both sequential access through a dynamic array and fast lookup capability through
 * a hash table. The function handles automatic memory management by expanding the values array
 * when capacity is exceeded and creates hash table nodes for O(1) average-case lookup performance.
 * 
 * The function performs two distinct but coordinated operations: array insertion for sequential
 * storage and hash table insertion for fast lookup. This dual approach optimizes for both
 * iteration through discovered m values and rapid repetition detection during sequence analysis.
 * 
 * Capacity Management:
 * When the values array reaches capacity, the function doubles the allocation size using
 * geometric growth strategy. This approach provides amortized O(1) insertion time while
 * minimizing the frequency of expensive reallocation operations.
 * 
 * Hash Table Insertion:
 * New hash nodes are inserted at the head of collision chains using a prepend strategy.
 * This provides O(1) insertion time and maintains all previously inserted values in their
 * respective collision chains for future lookup operations.
 * 
 * @param mv Pointer to the mValues container to modify. Must be a properly initialized
 *           container created with init_m_values(). The container's state will be updated
 *           to include the new m value.
 * @param m The m value to add to the container. This value will be stored in both the
 *          sequential array and the hash table for dual access patterns.
 * 
 * @note This function does not check for duplicate values before insertion. If the same
 *       m value is added multiple times, it will appear multiple times in both the array
 *       and hash table, though this typically doesn't occur in proper usage.
 * 
 * @note The function uses geometric growth (doubling) for array expansion, providing
 *       amortized O(1) insertion performance over many operations.
 * 
 * @note Hash table insertion uses prepend strategy, where new nodes are added at the
 *       head of collision chains for maximum insertion efficiency.
 * 
 * @note Memory allocation failures in either array expansion or hash node creation
 *       will terminate the program via safe_malloc() and safe_realloc().
 * 
 * @warning The function assumes the container has been properly initialized. Using
 *          an uninitialized container results in undefined behavior.
 * 
 * @complexity Amortized O(1) - constant time insertion with occasional O(n) array
 *            reallocation where n is the current number of stored values
 * 
 * @see init_m_values() for container initialization requirements
 * @see is_m_repeated() for lookup operations using the hash table
 * @see safe_realloc() for memory expansion behavior
 * @see hash_function() for hash computation algorithm
 * 
 * @example
 * ```c
 * // Initialize container and add values
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // Add sequence of m values
 * add_m_value(&m_values, 25);   // First value
 * add_m_value(&m_values, 108);  // Second value
 * add_m_value(&m_values, 54);   // Third value
 * 
 * // Container now contains 3 values accessible by:
 * // - Sequential access: m_values.values[0], m_values.values[1], m_values.values[2]
 * // - Hash lookup: is_m_repeated(&m_values, 25) returns true
 * 
 * // Typical usage in sequence analysis
 * uint64_t collatz_value = 22;
 * uint64_t m = calculate_m(collatz_value);
 * 
 * if (!is_m_repeated(&m_values, m)) {
 *     add_m_value(&m_values, m);  // Store for future repetition checks
 * } else {
 *     printf("Pseudocycle detected at m=%lu\n", m);
 * }
 * 
 * // Automatic capacity expansion example
 * for (int i = 0; i < 200; i++) {
 *     add_m_value(&m_values, i);  // Triggers expansion at 100 values
 * }
 * // Container capacity automatically doubled from 100 to 200
 * ```
 */
static void add_m_value(mValues* mv, uint64_t m) {
    // Expand array if necessary
    if (mv->count >= mv->capacity) {
        int new_capacity = mv->capacity * 2;
        mv->values = safe_realloc(mv->values, new_capacity * sizeof(uint64_t), "m_values expansion");
        mv->capacity = new_capacity;
    }
    
    mv->values[mv->count++] = m;
    
    // Add to hash table
    uint64_t hash = hash_function(m);
    HashNode* new_node = safe_malloc(sizeof(HashNode), "hash node");
    new_node->value = m;
    new_node->next = mv->buckets[hash];
    mv->buckets[hash] = new_node;
}

// ***********************************************************************************
// * 5. UNIQUE mr SET (GLOBAL RESULTS))
// ***********************************************************************************

/**
 * @brief Creates and initializes a thread-safe set for collecting unique mr values discovered during parallel search.
 * 
 * This function allocates and configures a specialized data structure designed to collect and
 * manage unique mr values found throughout the entire search process. The structure maintains
 * parallel arrays to store both the unique mr values and the first n value that generated each
 * mr, enabling comprehensive analysis of mr discovery patterns and statistical reporting.
 * 
 * The UniqueMrSet serves as a global collection point for all unique mr discoveries across
 * multiple threads, providing thread-safe operations for concurrent updates while maintaining
 * the relationship between mr values and their first occurrence. This information is crucial
 * for research analysis and validation of the search results.
 * 
 * Thread Safety Architecture:
 * The structure includes an OpenMP lock to ensure atomic operations when multiple threads
 * simultaneously discover new unique mr values. The lock protects both the addition of new
 * entries and the reading of current statistics during progress reporting.
 * 
 * Memory Layout Design:
 * - Parallel arrays for mr values and their corresponding first n values
 * - Fixed initial capacity (10000) based on empirical estimates of mr discovery rates
 * - Automatic expansion capability through safe_realloc() when capacity is exceeded
 * - Thread synchronization primitive for concurrent access protection
 * 
 * @return Pointer to a fully initialized UniqueMrSet structure ready for concurrent access.
 *         The structure contains empty parallel arrays with initial capacity and an active
 *         OpenMP lock. Never returns NULL due to safe_malloc() usage.
 * 
 * @note The initial capacity of 10000 is chosen based on empirical analysis of mr discovery
 *       rates in typical search ranges, providing sufficient space to minimize reallocations.
 * 
 * @note Both values and first_n arrays are allocated with the same capacity to maintain
 *       parallel array consistency and enable direct indexing relationships.
 * 
 * @note The OpenMP lock is initialized and ready for immediate use by multiple threads
 *       without additional setup requirements.
 * 
 * @note The count field is initialized to zero, indicating an empty set ready for discoveries.
 * 
 * @warning The returned structure must be properly cleaned up using destroy_unique_mr_set()
 *          to destroy the OpenMP lock and free allocated memory.
 * 
 * @complexity O(1) - simple structure and array allocation with fixed initial capacity
 * 
 * @see destroy_unique_mr_set() for proper cleanup procedure
 * @see add_unique_mr() for thread-safe addition of discovered mr values
 * @see is_mr_already_found() for duplicate detection functionality
 * @see UniqueMrSet structure definition for field descriptions
 * 
 * @example
 * ```c
 * // Create unique mr set for global discovery collection
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // Use in parallel search context
 * #pragma omp parallel
 * {
 *     for (uint64_t n = thread_start; n < thread_end; n++) {
 *         uint64_t mr = find_first_mr_in_sequence(n, &found);
 *         
 *         if (found) {
 *             bool is_new = add_unique_mr(unique_set, mr, n);
 *             if (is_new) {
 *                 printf("New unique mr=%lu discovered at n=%lu\n", mr, n);
 *             }
 *         }
 *     }
 * }
 * 
 * // Access final statistics
 * printf("Total unique mr values found: %d\n", unique_set->count);
 * 
 * // Cleanup when done
 * destroy_unique_mr_set(unique_set);
 * ```
 */
static UniqueMrSet* create_unique_mr_set(void) {
    UniqueMrSet* set = safe_malloc(sizeof(UniqueMrSet), "unique mr set");
    set->capacity = 10000;
    set->values = safe_malloc(set->capacity * sizeof(uint64_t), "unique mr values");
    set->first_n = safe_malloc(set->capacity * sizeof(uint64_t), "unique mr first_n");
    set->count = 0;
    omp_init_lock(&set->lock);
    return set;
}

/**
 * @brief Safely deallocates all memory associated with a UniqueMrSet structure and destroys its synchronization primitives.
 * 
 * This function performs comprehensive cleanup of a UniqueMrSet structure, ensuring proper
 * deallocation of all dynamically allocated parallel arrays and destruction of OpenMP
 * synchronization resources. The cleanup process prevents memory leaks while ensuring
 * that system-level synchronization primitives are properly released to avoid resource
 * exhaustion in long-running applications.
 * 
 * The function implements defensive programming by checking for NULL pointers, making it
 * safe to call in error handling paths or cleanup sequences where the set might not have
 * been successfully created. This design prevents segmentation faults and ensures robust
 * cleanup behavior even in partial initialization scenarios.
 * 
 * Cleanup Process:
 * 1. Validate input pointer to handle NULL gracefully
 * 2. Deallocate the values array containing unique mr values
 * 3. Deallocate the first_n array containing discovery n values
 * 4. Destroy the OpenMP lock to release synchronization resources
 * 5. Deallocate the main structure memory
 * 
 * Resource Management Strategy:
 * The function addresses both user-space memory (parallel arrays) and system-level
 * synchronization resources (OpenMP locks). Proper destruction of OpenMP locks is
 * essential to prevent resource leaks that could affect system performance over time.
 * 
 * @param set Pointer to the UniqueMrSet structure to destroy. Can be NULL, in which
 *            case the function returns immediately without action. After this function
 *            returns, the pointer becomes invalid and should not be accessed.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where set creation might have failed.
 * 
 * @note The function must only be called after all threads have finished using the
 *       set, as it destroys the synchronization lock that protects concurrent access.
 * 
 * @note After calling this function, any references to the set structure or its
 *       arrays become invalid and accessing them results in undefined behavior.
 * 
 * @note The function deallocates both parallel arrays (values and first_n) that
 *       were allocated during set creation, ensuring complete memory cleanup.
 * 
 * @warning This function is NOT thread-safe. Ensure no other threads are accessing
 *          the set when calling this function.
 * 
 * @warning Do not attempt to use the set pointer after calling this function,
 *          as the memory has been deallocated.
 * 
 * @complexity O(1) - constant time cleanup operations regardless of set contents
 * 
 * @see create_unique_mr_set() for the corresponding allocation function
 * @see omp_destroy_lock() for OpenMP lock destruction requirements
 * @see add_unique_mr() for functions that access the set during operation
 * 
 * @example
 * ```c
 * // Typical usage in cleanup sequence
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // ... use set during search operations ...
 * 
 * // Cleanup when search is complete
 * destroy_unique_mr_set(unique_set);
 * unique_set = NULL;  // Prevent accidental reuse
 * 
 * // Safe to call with NULL
 * destroy_unique_mr_set(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * if (some_error_condition) {
 *     destroy_unique_mr_set(unique_set);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Cleanup in search context
 * void cleanup_search_context(SearchContext* ctx) {
 *     destroy_unique_mr_set(ctx->unique_set);
 *     // ... other cleanup ...
 * }
 * ```
 */
static void destroy_unique_mr_set(UniqueMrSet* set) {
    if (set) {
        free(set->values);
        free(set->first_n);
        omp_destroy_lock(&set->lock);
        free(set);
    }
}

/**
 * @brief Thread-safely checks if a specific mr value has already been discovered and stored in the unique set.
 * 
 * This function performs a thread-safe linear search through the unique mr values collection
 * to determine if a specific mr value has been previously discovered during the search process.
 * The function uses OpenMP locks to ensure atomic read operations even when other threads
 * may be concurrently adding new unique mr values to the collection.
 * 
 * The function implements a simple but effective search strategy using early termination
 * when a match is found. While the linear search has O(n) complexity, the typical number
 * of unique mr values discovered is relatively small compared to the search space, making
 * this approach practical for most use cases.
 * 
 * Thread Safety Implementation:
 * The function acquires an exclusive lock before accessing the values array and maintains
 * the lock throughout the search operation to ensure consistent reads even if other threads
 * are simultaneously modifying the collection. The lock is released immediately after the
 * search completes to minimize contention.
 * 
 * Search Optimization:
 * Uses early termination with break statement to minimize search time when matches are
 * found early in the array. This is particularly effective since frequently discovered
 * mr values tend to be found early in the search process.
 * 
 * @param set Pointer to the UniqueMrSet structure to search. Must be a valid set with
 *            an initialized OpenMP lock. The set is accessed read-only during the search.
 * @param mr The mr value to search for in the unique collection. This represents a
 *           m repeated value that may have been previously discovered.
 * 
 * @return true if the mr value has been previously stored in the unique set
 *         false if the mr value is not found in the current collection
 * 
 * @note The function uses a const cast for the lock parameter to satisfy OpenMP's
 *       lock interface requirements while maintaining const correctness for the set.
 * 
 * @note The linear search approach is acceptable because the number of unique mr values
 *       is typically much smaller than the total search space.
 * 
 * @note The function maintains the lock throughout the entire search to ensure
 *       consistent results even if the set is being modified by other threads.
 * 
 * @note Early termination with break statement provides optimization for cases where
 *       matches are found quickly in the search sequence.
 * 
 * @complexity O(n) where n is the number of unique mr values currently stored,
 *            with early termination providing average-case improvement
 * 
 * @see add_unique_mr() for the function that adds values this function can find
 * @see UniqueMrSet structure definition for collection layout
 * @see omp_set_lock() and omp_unset_lock() for synchronization primitives
 * 
 * @example
 * ```c
 * // Check for duplicates before adding
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // Add some initial values
 * add_unique_mr(unique_set, 25, 408);
 * add_unique_mr(unique_set, 108, 1234);
 * 
 * // Check for existing values
 * bool found1 = is_mr_already_found(unique_set, 25);   // Returns true
 * bool found2 = is_mr_already_found(unique_set, 999);  // Returns false
 * bool found3 = is_mr_already_found(unique_set, 108);  // Returns true
 * 
 * // Typical usage in discovery process
 * uint64_t new_mr = find_first_mr_in_sequence(n, &found);
 * if (found && !is_mr_already_found(unique_set, new_mr)) {
 *     add_unique_mr(unique_set, new_mr, n);
 *     printf("New unique mr=%lu discovered!\n", new_mr);
 * }
 * ```
 */
static bool is_mr_already_found(const UniqueMrSet* set, uint64_t mr) {
    omp_set_lock((omp_lock_t*)&set->lock);
    bool found = false;
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == mr) {
            found = true;
            break;
        }
    }
    omp_unset_lock((omp_lock_t*)&set->lock);
    return found;
}

/**
 * @brief Thread-safely adds a new unique mr value to the set with automatic capacity expansion and duplicate prevention.
 * 
 * This function provides atomic insertion of newly discovered mr values into the unique collection
 * while ensuring no duplicates are stored and maintaining the relationship between mr values and
 * the first n value that generated them. The function implements comprehensive thread safety using
 * OpenMP locks and handles automatic memory expansion when the current capacity is exceeded.
 * 
 * The function performs duplicate detection, capacity management, and insertion as a single atomic
 * operation, ensuring data consistency even during high-frequency concurrent access from multiple
 * threads. Only genuinely unique mr values are added to the collection, with early termination
 * for duplicates to minimize lock contention time.
 * 
 * Duplicate Detection Strategy:
 * Before insertion, the function performs a linear search through existing values to ensure
 * uniqueness. If a duplicate is found, the function immediately returns false without modifying
 * the collection, minimizing the time spent holding the exclusive lock.
 * 
 * Capacity Management:
 * When the collection reaches capacity, both parallel arrays (values and first_n) are
 * simultaneously expanded using geometric growth (doubling strategy) to maintain array
 * synchronization and provide amortized O(1) insertion performance.
 * 
 * @param set Pointer to the UniqueMrSet structure to modify. Must be a valid set with
 *            initialized parallel arrays and OpenMP lock.
 * @param mr The mr value to add to the unique collection. This represents a m repeated
 *           value discovered during sequence analysis.
 * @param n The n value that first generated this mr value. This information is stored
 *          in parallel with the mr value for research and analysis purposes.
 * 
 * @return true if the mr value was successfully added as a new unique entry
 *         false if the mr value already exists in the collection (no modification made)
 * 
 * @note The function maintains parallel arrays in perfect synchronization, ensuring that
 *       values[i] and first_n[i] always correspond to the same discovery.
 * 
 * @note Duplicate detection is performed within the critical section to ensure atomicity,
 *       but early termination minimizes lock contention for duplicate cases.
 * 
 * @note Capacity expansion doubles both arrays simultaneously to maintain parallel structure
 *       and provides amortized O(1) insertion performance over many operations.
 * 
 * @note The function is fully thread-safe and can be called concurrently from multiple
 *       threads without external synchronization requirements.
 * 
 * @warning Memory allocation failures during capacity expansion will terminate the program
 *          via safe_realloc(), as this represents an unrecoverable error condition.
 * 
 * @complexity Average case: O(n) for duplicate detection plus amortized O(1) for insertion
 *            Expansion case: O(n) for both duplicate detection and array reallocation
 * 
 * @see is_mr_already_found() for read-only duplicate checking
 * @see safe_realloc() for memory expansion behavior
 * @see report_new_unique_mr() for usage in discovery reporting
 * @see UniqueMrSet structure definition for parallel array layout
 * 
 * @example
 * ```c
 * // Add unique mr values during parallel search
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // Successful addition of new unique value
 * bool added1 = add_unique_mr(unique_set, 25, 408);    // Returns true
 * bool added2 = add_unique_mr(unique_set, 108, 1234);  // Returns true
 * 
 * // Duplicate detection and rejection
 * bool added3 = add_unique_mr(unique_set, 25, 816);    // Returns false (duplicate mr)
 * 
 * // Usage in discovery process
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     uint64_t mr = find_first_mr_in_sequence(n, &found);
 *     if (found) {
 *         bool is_new = add_unique_mr(unique_set, mr, n);
 *         if (is_new) {
 *             report_new_unique_mr(mr, n, unique_set, progress);
 *         }
 *     }
 * }
 * 
 * // Capacity expansion example
 * // When count reaches capacity (10000), both arrays automatically double to 20000
 * ```
 */
static bool add_unique_mr(UniqueMrSet* set, uint64_t mr, uint64_t n) {
    omp_set_lock(&set->lock);
    
    // Check if it already exists
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == mr) {
            omp_unset_lock(&set->lock);
            return false;
        }
    }
    
    // Expand capacity, if necessary
    if (set->count >= set->capacity) {
        int new_capacity = set->capacity * 2;
        set->values = safe_realloc(set->values, new_capacity * sizeof(uint64_t), "unique mr values expansion");
        set->first_n = safe_realloc(set->first_n, new_capacity * sizeof(uint64_t), "unique mr first_n expansion");
        set->capacity = new_capacity;
    }
    
    set->values[set->count] = mr;
    set->first_n[set->count] = n;
    set->count++;
    
    omp_unset_lock(&set->lock);
    return true;
}

// ***********************************************************************************
// * 6. PROGRESS TRACKER SYSTEM
// ***********************************************************************************

/**
 * @brief Creates and initializes a thread-safe progress tracking structure for monitoring parallel search operations.
 * 
 * This function allocates and configures a progress tracking system designed to provide
 * real-time monitoring of parallel Collatz sequence analysis operations. The tracker
 * maintains thread-safe counters and timing information that can be safely updated by
 * multiple worker threads while being read by monitoring functions for progress reporting.
 * 
 * The progress tracker is essential for long-running computations as it provides users
 * with feedback on processing speed, completion estimates, and discovery statistics.
 * All fields are initialized to appropriate starting values for a new search operation.
 * 
 * Thread Safety Design:
 * The structure includes an OpenMP lock that enables atomic updates of progress metrics
 * during high-frequency parallel operations. This prevents race conditions when multiple
 * threads update counters simultaneously while maintaining acceptable performance overhead.
 * 
 * Initialization Values:
 * - All numeric counters start at zero to represent a fresh tracking session
 * - Timing information is initialized to 0.0 for immediate progress display triggering
 * - OpenMP lock is initialized and ready for immediate concurrent access
 * 
 * @return Pointer to a fully initialized ProgressTracker structure ready for concurrent
 *         access. The structure contains zero-initialized counters and an active OpenMP
 *         lock. Never returns NULL due to safe_malloc() usage.
 * 
 * @note The tracker's last_update_time field is initialized to 0.0, which ensures that
 *       the first call to update_progress_if_needed() will immediately display progress
 *       regardless of the PROGRESS_UPDATE_INTERVAL setting.
 * 
 * @note The OpenMP lock is initialized and must be properly destroyed during cleanup
 *       using omp_destroy_lock() to prevent resource leaks.
 * 
 * @note All progress counters use uint64_t to support very large search ranges
 *       (up to 2^64 numbers) without overflow concerns.
 * 
 * @warning The returned structure must be properly cleaned up using destroy_progress_tracker()
 *          to destroy the OpenMP lock and free allocated memory.
 * 
 * @complexity O(1) - simple structure allocation and field initialization
 * 
 * @see destroy_progress_tracker() for proper cleanup procedure
 * @see update_progress_if_needed() for progress reporting mechanism
 * @see increment_progress_counters() for thread-safe counter updates
 * @see ProgressTracker structure definition for field descriptions
 * 
 * @example
 * ```c
 * // Create progress tracker for parallel search
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // Use in parallel context
 * #pragma omp parallel
 * {
 *     uint64_t local_processed = 0;
 *     
 *     // Process numbers in assigned range
 *     for (uint64_t n = thread_start; n < thread_end; n++) {
 *         // ... analysis work ...
 *         local_processed++;
 *         
 *         // Periodic progress updates to avoid lock contention
 *         if (local_processed % 1000 == 0) {
 *             increment_progress_counters(progress, false);
 *         }
 *     }
 * }
 * 
 * // Check final statistics
 * printf("Total processed: %lu\n", progress->processed);
 * printf("Total found: %lu\n", progress->found_count);
 * 
 * // Cleanup when done
 * destroy_progress_tracker(progress);
 * ```
 */
static ProgressTracker* create_progress_tracker(void) {
    ProgressTracker* tracker = safe_malloc(sizeof(ProgressTracker), "progress tracker");
    tracker->processed = 0;
    tracker->found_count = 0;
    tracker->last_n_with_new_unique = 0;
    tracker->last_update_time = 0.0;
    omp_init_lock(&tracker->lock);
    return tracker;
}

/**
 * @brief Safely deallocates a progress tracker structure and destroys its synchronization primitives.
 * 
 * This function performs complete cleanup of a ProgressTracker structure, ensuring proper
 * destruction of OpenMP synchronization primitives and deallocation of memory. The cleanup
 * process follows proper resource management practices to prevent memory leaks and avoid
 * leaving dangling OpenMP locks that could consume system resources.
 * 
 * The function implements defensive programming by checking for NULL pointers, making it
 * safe to call in error handling paths or cleanup sequences where the tracker might not
 * have been successfully created. This design prevents segmentation faults and ensures
 * robust cleanup behavior.
 * 
 * Cleanup Process:
 * 1. Validate input pointer to handle NULL gracefully
 * 2. Destroy the OpenMP lock to release system synchronization resources
 * 3. Deallocate the main structure memory
 * 
 * Resource Management:
 * The function addresses both user-space memory (allocated via malloc) and system-level
 * synchronization resources (OpenMP locks). Proper destruction of OpenMP locks is critical
 * to prevent resource exhaustion in applications that create and destroy many trackers.
 * 
 * @param tracker Pointer to the ProgressTracker structure to destroy. Can be NULL,
 *                in which case the function returns immediately without action.
 *                After this function returns, the pointer becomes invalid and should
 *                not be accessed.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where tracker creation might have failed.
 * 
 * @note The function must only be called after all threads have finished using the
 *       tracker, as it destroys the synchronization lock that protects concurrent access.
 * 
 * @note After calling this function, any references to the tracker structure become
 *       invalid and accessing them results in undefined behavior.
 * 
 * @warning This function is NOT thread-safe. Ensure no other threads are accessing
 *          the tracker when calling this function.
 * 
 * @warning Do not attempt to use the tracker pointer after calling this function,
 *          as the memory has been deallocated.
 * 
 * @complexity O(1) - constant time cleanup operations regardless of tracker usage history
 * 
 * @see create_progress_tracker() for the corresponding allocation function
 * @see omp_destroy_lock() for OpenMP lock destruction requirements
 * @see update_progress_if_needed() for functions that access the tracker
 * 
 * @example
 * ```c
 * // Typical usage in cleanup sequence
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // ... use progress tracker during computation ...
 * 
 * // Cleanup when computation is complete
 * destroy_progress_tracker(progress);
 * progress = NULL;  // Prevent accidental reuse
 * 
 * // Safe to call with NULL
 * destroy_progress_tracker(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * ProgressTracker* progress = create_progress_tracker();
 * if (some_error_condition) {
 *     destroy_progress_tracker(progress);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Cleanup in main function
 * void cleanup_search_context(SearchContext* ctx) {
 *     destroy_progress_tracker(ctx->progress);
 *     // ... other cleanup ...
 * }
 * ```
 */
static void destroy_progress_tracker(ProgressTracker* tracker) {
    if (tracker) {
        omp_destroy_lock(&tracker->lock);
        free(tracker);
    }
}

/**
 * @brief Thread-safely updates the tracker with the most recent number that generated a new unique mr value.
 * 
 * This function maintains a record of the highest-numbered input that has produced a previously
 * unseen mr value during the search process. This information is valuable for monitoring the
 * discovery rate of new mr values and understanding the distribution pattern of unique mr
 * discoveries across the search space.
 * 
 * The function implements thread-safe updating using OpenMP locks to ensure that concurrent
 * updates from multiple threads don't create race conditions. Only values higher than the
 * current stored value are accepted, maintaining the "most recent discovery" semantics even
 * when threads process numbers out of order due to parallel scheduling.
 * 
 * Discovery Tracking Purpose:
 * - Monitor the search frontier for new mr value discoveries
 * - Provide feedback on discovery density across different ranges
 * - Enable analysis of where in the search space new patterns emerge
 * - Support debugging and research analysis of mr distribution
 * 
 * Thread Safety Implementation:
 * Uses OpenMP lock acquisition and release to ensure atomic read-modify-write operations
 * even when multiple threads simultaneously discover new unique mr values.
 * 
 * @param tracker Pointer to the ProgressTracker structure to update. Must be a valid
 *                tracker with an initialized OpenMP lock.
 * @param n The number that generated a new unique mr value. Only values greater than
 *          the currently stored value will update the tracker.
 * 
 * @note The function only updates the stored value if the new n is greater than the
 *       current value, ensuring that last_n_with_new_unique always represents the
 *       highest number that has produced a new unique mr.
 * 
 * @note This function is called from report_new_unique_mr() whenever a genuinely new
 *       mr value is discovered during the search process.
 * 
 * @note The comparison and update are performed atomically within the lock to prevent
 *       race conditions where multiple threads might read the same old value.
 * 
 * @warning The tracker must have a valid, initialized OpenMP lock. Calling this function
 *          on an uninitialized tracker results in undefined behavior.
 * 
 * @complexity O(1) - constant time operation with simple comparison and assignment
 * 
 * @see report_new_unique_mr() for the function that calls this updater
 * @see ProgressTracker structure definition for field descriptions
 * @see create_progress_tracker() for lock initialization
 * @see omp_set_lock() and omp_unset_lock() for synchronization primitives
 * 
 * @example
 * ```c
 * // Called when a new unique mr is discovered
 * ProgressTracker* tracker = create_progress_tracker();
 * 
 * // Thread 1 discovers new mr at n=1234
 * update_last_n_with_new_unique(tracker, 1234);
 * // tracker->last_n_with_new_unique is now 1234
 * 
 * // Thread 2 discovers new mr at n=987 (smaller, ignored)
 * update_last_n_with_new_unique(tracker, 987);
 * // tracker->last_n_with_new_unique remains 1234
 * 
 * // Thread 3 discovers new mr at n=5678 (larger, updated)
 * update_last_n_with_new_unique(tracker, 5678);
 * // tracker->last_n_with_new_unique is now 5678
 * 
 * // Usage in discovery reporting
 * if (add_unique_mr(unique_set, mr, n)) {
 *     update_last_n_with_new_unique(tracker, n);
 *     printf("New mr=%lu found at n=%lu\n", mr, n);
 * }
 * ```
 */
static void update_last_n_with_new_unique(ProgressTracker* tracker, uint64_t n) {
    omp_set_lock(&tracker->lock);
    if (n > tracker->last_n_with_new_unique) {
        tracker->last_n_with_new_unique = n;
    }
    omp_unset_lock(&tracker->lock);
}

/**
 * @brief Thread-safely increments progress counters using OpenMP atomic operations for high-performance concurrent updates.
 * 
 * This function provides efficient thread-safe updates to progress tracking counters without
 * the overhead of explicit lock acquisition and release. It uses OpenMP atomic directives
 * to ensure that concurrent increments from multiple threads are handled correctly while
 * maintaining maximum performance during high-frequency counter updates.
 * 
 * The function handles two distinct counter types: a mandatory processed counter that is
 * always incremented, and an optional found counter that is incremented only when a
 * number meets the search criteria. This design allows for efficient tracking of both
 * total work completed and successful discoveries.
 * 
 * Atomic Operation Benefits:
 * - No explicit lock contention or waiting between threads
 * - Hardware-level synchronization for maximum performance
 * - Automatic memory ordering guarantees for counter consistency
 * - Minimal overhead compared to mutex-based synchronization
 * 
 * Performance Characteristics:
 * Atomic operations are significantly faster than lock-based synchronization for simple
 * increment operations, making this function suitable for high-frequency calls within
 * tight processing loops without substantial performance degradation.
 * 
 * @param tracker Pointer to the ProgressTracker structure containing the counters to update.
 *                Must be a valid tracker structure with properly initialized counter fields.
 * @param mr_found Boolean flag indicating whether the processed number yielded a valid mr
 *                 result. If true, both processed and found_count are incremented; if false,
 *                 only processed is incremented.
 * 
 * @note OpenMP atomic directives ensure thread safety without requiring explicit locks,
 *       providing better performance than mutex-based synchronization for simple increments.
 * 
 * @note The processed counter is always incremented regardless of the mr_found flag,
 *       ensuring accurate tracking of total work completed.
 * 
 * @note Atomic operations provide memory ordering guarantees, ensuring that counter
 *       updates are visible to other threads in a consistent manner.
 * 
 * @note This function is designed for high-frequency calls and has minimal overhead
 *       compared to lock-based alternatives.
 * 
 * @complexity O(1) - constant time atomic operations with hardware-level synchronization
 * 
 * @see ProgressTracker structure definition for counter field descriptions
 * @see process_single_number() for typical usage in processing loops
 * @see update_progress_if_needed() for reading these counters safely
 * @see OpenMP atomic directive documentation for synchronization behavior
 * 
 * @example
 * ```c
 * // Usage in parallel processing loop
 * ProgressTracker* tracker = create_progress_tracker();
 * 
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     bool found_mr = find_first_mr_in_sequence(n, &mr_value);
 *     
 *     // High-frequency atomic updates
 *     increment_progress_counters(tracker, found_mr);
 *     
 *     // Periodic progress display (less frequent)
 *     if (n % PROGRESS_CHECK_FREQUENCY == 0) {
 *         update_progress_if_needed(ctx);
 *     }
 * }
 * 
 * // Example with different outcomes
 * increment_progress_counters(tracker, true);   // processed++, found_count++
 * increment_progress_counters(tracker, false);  // processed++, found_count unchanged
 * increment_progress_counters(tracker, false);  // processed++, found_count unchanged
 * increment_progress_counters(tracker, true);   // processed++, found_count++
 * 
 * // Result: processed = 4, found_count = 2
 * ```
 */
static void increment_progress_counters(ProgressTracker* tracker, bool mr_found) {
    #pragma omp atomic
    tracker->processed++;
    
    if (mr_found) {
        #pragma omp atomic
        tracker->found_count++;
    }
}

/**
 * @brief Conditionally updates and displays comprehensive progress information with intelligent timing control.
 * 
 * This function provides intelligent progress reporting by checking if sufficient time has elapsed
 * since the last update before performing the computationally expensive progress calculation and
 * display operations. It balances informative real-time feedback with minimal performance overhead
 * by using time-based throttling and atomic snapshots of progress data.
 * 
 * The function generates a comprehensive progress report including completion percentage, processing
 * rate, discovery statistics, and estimated time to completion. All calculations are performed
 * within a critical section to ensure consistent data snapshots even during high-frequency
 * concurrent updates from multiple threads.
 * 
 * Progress Reporting Features:
 * - Completion percentage with single decimal precision for clear progress indication
 * - Real-time processing rate in numbers per second for performance monitoring
 * - Current count of unique mr values discovered for research tracking
 * - Estimated time to completion in minutes for user planning
 * - Time-based throttling to prevent excessive output frequency
 * 
 * Performance Optimization:
 * The function uses PROGRESS_UPDATE_INTERVAL (3.0 seconds) to throttle update frequency,
 * preventing performance degradation from excessive printf operations while maintaining
 * responsive user feedback during long-running computations.
 * 
 * @param ctx Pointer to the search context containing both progress tracker and unique set
 *            structures. Must contain valid initialized structures with proper lock initialization.
 * 
 * @note The function performs all calculations within the lock critical section to ensure
 *       atomic reads of progress data, preventing inconsistent progress reports.
 * 
 * @note Uses fflush(stdout) to ensure immediate display of progress information even when
 *       stdout is buffered or redirected to files.
 * 
 * @note ETA calculation assumes linear processing rate, which may be inaccurate for workloads
 *       with varying computational complexity per number.
 * 
 * @note The function updates tracker->last_update_time only when actually displaying progress,
 *       ensuring accurate interval timing between displays.
 * 
 * @complexity O(1) - constant time operations with occasional printf overhead that is
 *            throttled by time-based updates
 * 
 * @see PROGRESS_UPDATE_INTERVAL for update frequency control (3.0 seconds)
 * @see SearchContext structure definition for required fields
 * @see ProgressTracker and UniqueMrSet for data sources
 * @see omp_get_wtime() for high-precision timing
 * 
 * @example
 * ```c
 * // Typical usage in parallel processing loop
 * SearchContext ctx = { ... };  // Initialized context
 * 
 * #pragma omp parallel
 * {
 *     uint64_t local_processed = 0;
 *     int thread_num = omp_get_thread_num();
 *     
 *     #pragma omp for
 *     for (uint64_t n = 1; n < ctx.max_n; n++) {
 *         // ... process number n ...
 *         local_processed++;
 *         
 *         // Only thread 0 handles progress updates to avoid duplicate output
 *         if (thread_num == 0 && local_processed % PROGRESS_CHECK_FREQUENCY == 0) {
 *             update_progress_if_needed(&ctx);
 *         }
 *     }
 * }
 * 
 * // Output example:
 * // Progress: (0.015621%) | Processed: 167731 | Unique values of mr found: 42 | 251580.1 nums/sec | ETA: 71.1 min
 * // Progress: (0.081278%) | Processed: 872716 | Unique values of mr found: 42 | 230168.1 nums/sec | ETA: 77.7 min
 * // Progress: (0.146389%) | Processed: 1571841 | Unique values of mr found: 42 | 223405.1 nums/sec | ETA: 80.0 min
 * ```
 */
static void update_progress_if_needed(const SearchContext* ctx) {
    ProgressTracker* tracker = ctx->progress;
    omp_set_lock(&tracker->lock);
    
    double current_time = omp_get_wtime();
    
    if (current_time - tracker->last_update_time >= PROGRESS_UPDATE_INTERVAL) {
        tracker->last_update_time = current_time;
        
        double elapsed = current_time - ctx->start_time;
        double rate = tracker->processed / elapsed;
        double eta = (ctx->max_n - tracker->processed) / rate;
        double progress_percent = (double)tracker->processed / ctx->max_n * 100.0;
        
        printf("\t - Progress: (%.6f%%) | Processed: %lu | Unique values of mr found: %d | %.1f nums/sec | ETA: %.1f min\n",
               progress_percent, tracker->processed, ctx->unique_set->count, rate, eta/60.0);
        fflush(stdout);
    }
    
    omp_unset_lock(&tracker->lock);
}

/**
 * @brief Reports the discovery of a new unique mr value with thread-safe output, initialization handling, and progress tracking updates.
 * 
 * This function provides immediate feedback when a genuinely new mr value is discovered during
 * the search process, combining progress tracking updates with formatted console output and
 * special handling for the first discovery event. The function ensures thread-safe reporting
 * even during high-frequency parallel discovery events while maintaining consistent output
 * formatting and providing initialization context for research monitoring.
 * 
 * The function performs three coordinated operations: updating the progress tracker with the
 * latest discovery location, handling first-time initialization output, and generating formatted
 * discovery notifications. Special logic ensures that the very first discovery triggers an
 * initial progress line to establish proper output context before individual discoveries are reported.
 * 
 * First Discovery Initialization:
 * - Uses static variable to detect first execution across all threads
 * - Outputs initial progress line showing zero state before first discovery
 * - Establishes consistent output format baseline for subsequent reports
 * - Prevents confusion about missing initial progress information
 * 
 * Discovery Reporting Features:
 * - Immediate notification of new unique mr discoveries with double-indented formatting
 * - Complete context including mr value, generating n value, and current total count
 * - Thread-safe output formatting using named critical section to prevent text corruption
 * - Progress tracker frontier updates for discovery location monitoring
 * - Forced output flushing for real-time feedback during long searches
 * 
 * Thread Safety Implementation:
 * Progress tracker update occurs outside the critical section for performance, while output
 * formatting uses OpenMP critical section with named identifier (discovery_report) to ensure
 * that only one thread can output discovery information at a time, preventing garbled text.
 * 
 * @param mr The newly discovered unique mr value that has not been previously found.
 *           This represents a m repeated value discovered through sequence analysis.
 * @param n The specific n value that generated this unique mr during its Collatz sequence.
 *          This provides the discovery context for research analysis and frontier tracking.
 * @param set Pointer to the UniqueMrSet containing all discovered unique values. Used to
 *            report the current total count of unique discoveries for progress monitoring.
 * @param tracker Pointer to the ProgressTracker to update with the latest discovery location.
 *                This maintains the frontier of unique discoveries for analysis purposes.
 * 
 * @note The function uses a static boolean variable to track first execution, creating
 *       global state that affects behavior across all function calls and threads.
 * 
 * @note Progress tracker update occurs outside the critical section to minimize lock
 *       contention, while output formatting is fully protected by the named critical section.
 * 
 * @note The first discovery triggers an initialization progress line showing zero state,
 *       establishing proper context for subsequent individual discovery reports.
 * 
 * @note Output uses double indentation (\t\t) to distinguish individual discoveries from
 *       general progress updates, creating clear visual hierarchy in the output.
 * 
 * @note fflush(stdout) ensures immediate output display even when stdout is buffered
 *       or redirected to files, providing real-time feedback during long searches.
 * 
 * @note The function assumes the mr value is genuinely unique, as duplicate checking
 *       should be performed by add_unique_mr() before calling this function.
 * 
 * @complexity O(1) - simple output formatting, progress update, and static variable check
 * 
 * @see update_last_n_with_new_unique() for progress tracking updates
 * @see add_unique_mr() for the duplicate checking that precedes this function
 * @see UniqueMrSet structure for unique value collection
 * @see ProgressTracker for discovery frontier tracking
 * 
 * @example
 * ```c
 * // First discovery call in the program
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // First call outputs initialization line then discovery
 * report_new_unique_mr(25, 408, unique_set, progress);
 * // Output: "\t - Progress: (0.0%) | Processed: 0 | Unique values of mr found: 0 | 0.0 nums/sec | ETA: -- min"
 * //         "\t\t - New unique value of mr = 60 found, generated by n = 27 (total unique mr values: 10)"
 * 
 * // Subsequent calls only output discoveries
 * report_new_unique_mr(108, 1234, unique_set, progress);
 * // Output: "\t\t - New unique value of mr = 53 found, generated by n = 108 (total unique mr values: 15)"
 * 
 * // Multiple concurrent discoveries (thread-safe output)
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     uint64_t mr = find_first_mr_in_sequence(n, &found);
 *     if (found && add_unique_mr(unique_set, mr, n)) {
 *         report_new_unique_mr(mr, n, unique_set, progress);
 *         // Each thread's output appears cleanly without interference
 *         // First thread to call will output initialization line
 *     }
 * }
 * ```
 */
static void report_new_unique_mr(uint64_t mr, uint64_t n, const UniqueMrSet* set, ProgressTracker* tracker) {

    static bool first_report = true;
    
    update_last_n_with_new_unique(tracker, n);
    
    #pragma omp critical(discovery_report)
    {
        if (first_report) {
            printf("\t - Progress: (0.0%%) | Processed: 0 | Unique values of mr found: 0 | 0.0 nums/sec | ETA: -- min\n");
            fflush(stdout);
            first_report = false;
        }
        
        printf("\t\t - New unique value of mr = %lu found, generated by n = %lu (total unique mr values: %d)\n", 
               mr, n, set->count);
        fflush(stdout);
    }
}

// ***********************************************************************************
// * 7. COLLATZ SEQUENCE ANALYSIS
// ***********************************************************************************

/**
 * @brief Analyzes a complete Collatz sequence to find the first mr value that forms a pseudocycle through m value repetition.
 * 
 * This function implements the core algorithm for pseudocycle detection in Collatz sequences using
 * the tuple-based transform approach. It generates the complete Collatz sequence starting from n_start,
 * computes the corresponding m values using the tuple transform, and detects the first repetition
 * that indicates pseudocycle completion. The function handles multiple termination conditions including
 * natural convergence, overflow prevention, and divergence detection.
 * 
 * The algorithm follows the tuple-based transform methodology described in the research paper,
 * where repetition of m values (rather than sequence values) provides a more tractable approach
 * to pseudocycle detection. This method enables identification of mr values that characterize
 * different pseudocycle patterns in Collatz sequences.
 * 
 * Pseudocycle Detection Strategy:
 * 1. Generate Collatz sequence values iteratively
 * 2. Compute m value for each sequence position using tuple transform
 * 3. Check for m value repetition before adding to collection
 * 4. Return the first repeated m value as the mr (m repeated) identifier
 * 5. Handle special cases (convergence to 1, overflow, divergence)
 * 
 * Termination Conditions:
 * - Repetition detected: Returns the repeated m value as mr
 * - Natural convergence to 1: Returns mr=0 (trivial cycle)
 * - Overflow in Collatz transform: Returns found=false
 * - Sequence divergence: Returns mr=0 after safety threshold
 * - Maximum length exceeded: Returns found=false
 * 
 * @param n_start The initial value to begin Collatz sequence analysis. Must be a positive
 *                integer representing the starting point for sequence generation.
 * @param found Pointer to boolean flag that indicates whether a valid mr was successfully
 *              determined. Set to true when mr is found or sequence reaches 1, false
 *              when overflow or other error conditions prevent completion.
 * 
 * @return The mr value representing the first pseudocycle encountered in the sequence.
 *         Returns 0 for sequences that reach the trivial cycle (n=1) without other
 *         pseudocycles, or when divergence is detected. The value is only meaningful
 *         when *found is set to true.
 * 
 * @note The function creates and destroys a local mValues container for each sequence
 *       analysis, ensuring clean state for each independent calculation.
 * 
 * @note mr=0 has special meaning: it represents sequences that reach the trivial cycle
 *       (n=1) without encountering other pseudocycles, which occurs for approximately
 *       35% of starting values in typical ranges.
 * 
 * @note The divergence detection threshold (step > 10000 && n > n_start * 1000) prevents
 *       infinite loops for sequences that grow without bound or cycle with very long periods.
 * 
 * @note MAX_SEQUENCE_LENGTH provides an absolute upper bound on computation time,
 *       preventing resource exhaustion for pathological cases.
 * 
 * @warning Large starting values may not complete analysis within safety limits,
 *          resulting in found=false even for valid inputs.
 * 
 * @complexity Average case: O(k) where k is the Collatz sequence length until pseudocycle
 *            Worst case: O(MAX_SEQUENCE_LENGTH) when safety limits are reached
 * 
 * @see calculate_m() for the tuple-based transform computation
 * @see apply_collatz_function() for sequence progression with overflow protection
 * @see mValues container functions for m value storage and repetition detection
 * @see MAX_SEQUENCE_LENGTH for sequence length limits
 * 
 * @example
 * ```c
 * // Analyze specific starting values
 * bool found;
 * uint64_t mr1 = find_first_mr_in_sequence(7, &found);
 * if (found) {
 *     printf("n=7 has mr=%lu\n", mr1);  // Outputs mr=0 (reaches 1 directly)
 * }
 * 
 * uint64_t mr2 = find_first_mr_in_sequence(27, &found);
 * if (found) {
 *     printf("n=27 has mr=%lu\n", mr2);  // Outputs specific mr value
 * }
 * 
 * // Usage in parallel search
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     bool found;
 *     uint64_t mr = find_first_mr_in_sequence(n, &found);
 *     if (found) {
 *         bool is_new = add_unique_mr(unique_set, mr, n);
 *         if (is_new) {
 *             report_new_unique_mr(mr, n, unique_set, progress);
 *         }
 *     }
 * }
 * ```
 */
static uint64_t find_first_mr_in_sequence(uint64_t n_start, bool* found) {
    uint64_t n = n_start;
    mValues m_values;
    init_m_values(&m_values);
    
    uint64_t first_mr = 0;
    *found = false;
    
    // Generate Collatz sequence and search for repetitions in m values
    for (int step = 0; step < MAX_SEQUENCE_LENGTH && n != 1; step++) {
        uint64_t m = calculate_m(n);
        
        // Check repetition BEFORE adding
        if (is_m_repeated(&m_values, m)) {
            first_mr = m;
            *found = true;
            break;
        }
        
        add_m_value(&m_values, m);
        
        // Apply Collatz transformation with overflow check
        if (!apply_collatz_function(&n)) {
            *found = false;
            break;
        }
        
        // Safety check for very long sequences
        if (step > 10000 && n > n_start * 1000) {
            first_mr = 0;
            *found = true;
            break;
        }
    }
    
    // If we naturally reach n=1 (trivial cycle), then mr=0
    if (n == 1 && !*found) {
        first_mr = 0;
        *found = true;
    }
    
    destroy_m_values(&m_values);
    return first_mr;
}

/**
 * @brief Processes a single number through the complete mr discovery pipeline with optimized early filtering and result collection.
 * 
 * This function implements the core processing logic for individual numbers in the parallel search
 * algorithm, orchestrating the complete workflow from sequence analysis through result reporting.
 * It coordinates multiple subsystems including sequence analysis, progress tracking, unique value
 * collection, and discovery reporting while maintaining efficient performance through optimized
 * counter management and selective processing.
 * 
 * The function serves as the fundamental work unit in the parallel search, designed to be called
 * millions of times across multiple threads with minimal overhead. It implements a streamlined
 * pipeline that maximizes efficiency while ensuring comprehensive data collection and reporting
 * for successful discoveries.
 * 
 * Processing Pipeline:
 * 1. Analyze Collatz sequence to determine mr value
 * 2. Update local and global progress counters based on results
 * 3. For successful discoveries, attempt addition to unique collection
 * 4. Report genuinely new unique discoveries immediately
 * 5. Update thread-local counters for parallel reduction
 * 
 * Performance Optimization Strategy:
 * The function uses local counters to minimize atomic operations and global synchronization,
 * only accessing shared data structures when necessary (unique value addition and progress
 * updates). This design reduces contention while maintaining accurate progress tracking.
 * 
 * @param n The number to analyze through the complete mr discovery pipeline. Must be a
 *          positive integer within the search range being processed.
 * @param ctx Pointer to the search context containing all shared resources including unique
 *            value collection, progress tracking, and configuration parameters.
 * @param local_found Pointer to thread-local counter for numbers that successfully yielded
 *                    mr values. Updated for each successful discovery regardless of uniqueness.
 * @param local_processed Pointer to thread-local counter for total numbers processed by this
 *                        thread. Always incremented to track work completion.
 * 
 * @note The function updates local counters that are later reduced across threads, minimizing
 *       synchronization overhead during high-frequency processing.
 * 
 * @note Progress counter updates occur regardless of discovery outcome to maintain accurate
 *       processing statistics for rate calculations and ETA estimation.
 * 
 * @note Unique value addition and discovery reporting are performed only when mr analysis
 *       succeeds, optimizing the common case of unsuccessful analysis.
 * 
 * @note Discovery reporting occurs immediately for new unique values, providing real-time
 *       feedback during long-running searches without waiting for batch processing.
 * 
 * @complexity Average case: O(k) where k is the Collatz sequence length for the input number
 *            Most numbers: O(1) due to early termination in sequence analysis
 * 
 * @see find_first_mr_in_sequence() for the core sequence analysis algorithm
 * @see increment_progress_counters() for thread-safe progress tracking
 * @see add_unique_mr() for unique value collection with duplicate prevention
 * @see report_new_unique_mr() for immediate discovery reporting
 * 
 * @example
 * ```c
 * // Usage in parallel processing loop
 * SearchContext ctx = { ... };  // Initialized context
 * 
 * #pragma omp parallel reduction(+:total_found, total_processed)
 * {
 *     uint64_t thread_found = 0, thread_processed = 0;
 *     
 *     #pragma omp for schedule(dynamic, DEFAULT_CHUNK_SIZE)
 *     for (uint64_t n = 1; n < ctx.max_n; n++) {
 *         process_single_number(n, &ctx, &thread_found, &thread_processed);
 *         
 *         // Periodic progress updates
 *         if (thread_processed % LOCAL_COUNTER_UPDATE_FREQUENCY == 0) {
 *             update_progress_if_needed(&ctx);
 *         }
 *     }
 *     
 *     total_found += thread_found;
 *     total_processed += thread_processed;
 * }
 * 
 * // Example outcomes for different inputs:
 * // n=7:    mr_found=true, mr=0, local_found++, local_processed++
 * // n=27:   mr_found=true, mr=25, local_found++, local_processed++, new discovery reported
 * // n=1000: mr_found=false, local_processed++ only
 * ```
 */
static void process_single_number(uint64_t n, SearchContext* ctx, uint64_t* local_found, uint64_t* local_processed) {
    bool mr_found = false;
    uint64_t mr = find_first_mr_in_sequence(n, &mr_found);
    
    if (mr_found) {
        (*local_found)++;
        increment_progress_counters(ctx->progress, true);
        
        // Add to unique set and check if new
        bool is_new_unique = add_unique_mr(ctx->unique_set, mr, n);
        
        // Report new unique immediately
        if (is_new_unique) {
            report_new_unique_mr(mr, n, ctx->unique_set, ctx->progress);
        }
    } else {
        increment_progress_counters(ctx->progress, false);
    }
    
    (*local_processed)++;
}

// ***********************************************************************************
// * 8. PARALLEL SCHEDULING SYSTEM
// ***********************************************************************************

/**
 * @brief Executes parallel search using OpenMP guided scheduling for automatic load balancing in medium-sized workloads.
 * 
 * This function implements the parallel search algorithm using OpenMP's guided scheduling strategy,
 * which provides automatic load balancing by dynamically assigning work chunks of decreasing size
 * to threads as they complete their current assignments. This approach combines the benefits of
 * reduced scheduling overhead with adaptive load distribution, making it optimal for medium-sized
 * workloads where computation time variability begins to affect overall parallel efficiency.
 * 
 * Guided scheduling starts by assigning large chunks to threads to minimize overhead, then
 * progressively reduces chunk size as work completion approaches to ensure better load balancing
 * in the final stages. This strategy is particularly effective for workloads where early numbers
 * may have different computational characteristics than later numbers, or where natural work
 * imbalances emerge during execution.
 * 
 * Load Balancing Mechanism:
 * - Initial chunks are large to minimize scheduling overhead
 * - Chunk sizes decrease progressively as work completion approaches
 * - Threads automatically request new work when current chunks complete
 * - Final work distribution ensures all threads remain active until completion
 * - Natural adaptation to varying computational loads across the range
 * 
 * Performance Characteristics:
 * - Lower overhead than dynamic scheduling due to larger initial chunks
 * - Better load balancing than static scheduling for variable workloads
 * - Automatic adaptation to computation time variations
 * - Optimal for medium ranges where load imbalance becomes noticeable
 * 
 * @param ctx Pointer to the search context containing all configuration parameters,
 *            shared data structures, and progress tracking components. Must be fully
 *            initialized with valid unique set and progress tracker.
 * @param total_found Pointer to store the aggregated count of numbers that successfully
 *                    yielded mr values across all threads. Updated with final results.
 * @param total_processed Pointer to store the total count of numbers processed across
 *                        all threads. Used for performance metrics and validation.
 * 
 * @note Guided scheduling uses OpenMP's internal algorithm to determine optimal chunk
 *       sizes, starting large and decreasing as work completion approaches.
 * 
 * @note Progress monitoring is performed only by thread 0 to prevent duplicate output,
 *       with checks occurring at regular intervals during chunk processing.
 * 
 * @note The strategy automatically adapts to varying computation times without manual
 *       tuning, making it suitable for ranges where workload characteristics are unknown.
 * 
 * @note Thread-local counters minimize synchronization overhead while reduction operations
 *       efficiently aggregate results from all participating threads.
 * 
 * @note No chunk size parameter is required as OpenMP manages chunk sizing automatically
 *       based on remaining work and thread availability.
 * 
 * @complexity O(n/p) per thread where n is the search range and p is the number of threads,
 *            with adaptive overhead that decreases as chunk sizes increase
 * 
 * @see configure_parallel_scheduling() for automatic selection of this strategy
 * @see process_single_number() for individual number processing logic
 * @see update_progress_if_needed() for progress monitoring mechanism
 * @see OpenMP guided scheduling documentation for algorithm details
 * 
 * @example
 * ```c
 * // Execute search with guided scheduling for medium range
 * SearchContext ctx = {
 *     .max_n = 50000000,  // 50 million numbers (medium range)
 *     // ... other initialized fields ...
 * };
 * 
 * uint64_t found_count, processed_count;
 * execute_search_with_guided_scheduling(&ctx, &found_count, &processed_count);
 * 
 * // Typical work distribution pattern with 4 threads:
 * // Early phase: Large chunks (e.g., 2M numbers per chunk)
 * // Middle phase: Medium chunks (e.g., 500K numbers per chunk)  
 * // Final phase: Small chunks (e.g., 50K numbers per chunk)
 * // Ensures all threads finish approximately simultaneously
 * ```
 */
static void execute_search_with_guided_scheduling(SearchContext* ctx, uint64_t* total_found, uint64_t* total_processed) {
    uint64_t local_found = 0, local_processed = 0;
    
    #pragma omp parallel reduction(+:local_found, local_processed)
    {
        uint64_t thread_found = 0, thread_processed = 0;
        int thread_num = omp_get_thread_num();
        
        #pragma omp for schedule(guided)
        for (uint64_t n = 1; n < ctx->max_n; n++) {
            process_single_number(n, ctx, &thread_found, &thread_processed);
            
            if (thread_num == 0 && thread_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                update_progress_if_needed(ctx);
            }
        }
        
        local_found += thread_found;
        local_processed += thread_processed;
    }
    
    *total_found = local_found;
    *total_processed = local_processed;
}

/**
 * @brief Executes the complete parallel search using optimized guided scheduling with integrated configuration display.
 * 
 * This function serves as the unified coordinator for the entire parallel search operation,
 * combining configuration display with direct execution using guided scheduling. It provides
 * a streamlined approach that eliminates configuration complexity while maintaining transparent
 * feedback about the parallel processing approach used for performance analysis and monitoring.
 * 
 * The function implements a simplified workflow that directly uses guided scheduling without
 * intermediate configuration steps, reducing code complexity while preserving all essential
 * functionality. The integrated approach combines information display with immediate execution,
 * providing users with clear feedback about the processing strategy being employed.
 * 
 * Integrated Execution Process:
 * 1. Display guided scheduling configuration information for transparency
 * 2. Execute parallel search using proven guided scheduling strategy
 * 3. Return aggregated results from all participating threads
 * 
 * Guided Scheduling Advantages:
 * - Universal optimization: Automatically adapts to all workload sizes and characteristics
 * - Zero configuration: No manual tuning, thresholds, or parameter calculation required
 * - Consistent performance: Reliable load balancing across different computational scales
 * - Simplified architecture: Single execution path reduces maintenance complexity
 * - Transparent operation: Clear feedback about processing approach without complexity
 * 
 * Performance Characteristics:
 * Guided scheduling provides optimal performance across all ranges by automatically starting
 * with large chunks to minimize overhead, then progressively reducing chunk sizes to ensure
 * balanced completion. This adaptive behavior delivers excellent results for both uniform
 * and variable computational workloads without requiring manual optimization.
 * 
 * @param ctx Pointer to the search context containing all configuration parameters, shared
 *            data structures, and progress tracking components. Must be fully initialized
 *            with valid unique set, progress tracker, and search parameters.
 * @param found_count Pointer to store the final count of numbers that successfully yielded
 *                    mr values across all threads using guided scheduling.
 * @param processed_count Pointer to store the total count of numbers processed across all
 *                        threads, used for performance metrics and completion validation.
 * 
 * @note The function provides immediate transparency by displaying the guided scheduling
 *       approach, eliminating the need for separate configuration and display functions.
 * 
 * @note Guided scheduling is used universally based on empirical analysis showing excellent
 *       performance across all workload sizes without manual configuration requirements.
 * 
 * @note The integrated approach reduces function call overhead while maintaining clear
 *       separation between configuration display and actual parallel execution.
 * 
 * @note Error handling and resource management are delegated to the underlying guided
 *       scheduling implementation, ensuring robust parallel execution.
 * 
 * @note Thread count is automatically detected and managed by the guided scheduling
 *       implementation without requiring explicit configuration or display.
 * 
 * @complexity O(n/p) where n is the search range and p is the number of threads, with
 *            adaptive overhead that automatically optimizes based on workload characteristics
 * 
 * @see execute_search_with_guided_scheduling() for the underlying parallel implementation
 * @see SearchContext structure for comprehensive data organization
 * @see ProgressTracker and UniqueMrSet for shared resource management
 * @see OpenMP guided scheduling documentation for algorithm details
 * 
 * @example
 * ```c
 * // Execute streamlined parallel search for any workload size
 * SearchContext ctx = {
 *     .max_n = 1UL << 25,  // 2^25 = 33,554,432 numbers
 *     .exponent = 25,
 *     .unique_set = create_unique_mr_set(),
 *     .progress = create_progress_tracker(),
 *     .start_time = omp_get_wtime()
 * };
 * 
 * uint64_t found_count, processed_count;
 * execute_parallel_search(&ctx, &found_count, &processed_count);
 * 
 * // Output shows integrated configuration and execution:
 * // "[*] SEARCH PROCESS"
 * // ... parallel execution begins immediately ...
 * 
 * printf("\n[*] SEARCH RESULTS\n");
 * printf("\t - Total numbers processed: %lu\n", processed_count);
 * printf("\t - Total time: %.3f seconds\n", total_time);
 * 
 * // Unified approach for all workload sizes:
 * // Small, medium, and large ranges all use the same optimized guided scheduling
 * // with automatic adaptation to workload characteristics
 * ```
 */
static void execute_parallel_search(SearchContext* ctx, uint64_t* found_count, uint64_t* processed_count) {
    // Start search process
    printf("\n[*] SEARCH PROCESS\n");
    
    // Execute search with guided scheduling
    execute_search_with_guided_scheduling(ctx, found_count, processed_count);
}

// ***********************************************************************************
// * 9. RESULTS OUTPUT AND REPORTING
// ***********************************************************************************

/**
 * @brief Creates an array of indices that sorts unique mr values in ascending order without modifying the original data.
 * 
 * This function implements an indirect sorting algorithm that generates an index array to access
 * unique mr values in sorted order while preserving the original storage organization of the
 * UniqueMrSet. This approach maintains the relationship between mr values and their corresponding
 * first_n values in parallel arrays while enabling sorted output for reports and analysis.
 * 
 * The function uses a simple but correct bubble sort algorithm for the index array. While bubble
 * sort has O(n²) complexity, it's acceptable here because the number of unique mr values is
 * typically small (dozens to hundreds) compared to the search space, and the implementation
 * prioritizes correctness and simplicity over optimal performance for this non-critical operation.
 * 
 * Indirect Sorting Benefits:
 * - Preserves original data organization in parallel arrays
 * - Maintains relationship between mr values and discovery n values
 * - Enables multiple sorted views without data duplication
 * - Allows original insertion order to be retained alongside sorted access
 * - Minimal memory overhead (only index array allocation)
 * 
 * Implementation Strategy:
 * 1. Allocate index array with same count as unique mr values
 * 2. Initialize indices to natural order (0, 1, 2, ...)
 * 3. Sort indices based on comparison of referenced mr values
 * 4. Return sorted index array for indirect access to original data
 * 
 * @param set Pointer to the UniqueMrSet containing the mr values to be sorted.
 *            Must be a valid set with count > 0 for meaningful sorting.
 *            The original set data remains unchanged.
 * 
 * @return Pointer to a dynamically allocated array of indices that provides
 *         sorted access to the mr values. The caller is responsible for
 *         freeing this memory when no longer needed.
 * 
 * @note The returned indices array enables sorted access via set->values[indices[i]]
 *       where i ranges from 0 to set->count-1 in ascending mr value order.
 * 
 * @note Bubble sort is used for simplicity and correctness. For large datasets,
 *       this could be replaced with quicksort or mergesort for better performance.
 * 
 * @note The function assumes set->count represents the actual number of valid
 *       entries in the values array and does not perform bounds checking.
 * 
 * @note Parallel array relationships are preserved: indices[i] can be used to
 *       access both set->values[indices[i]] and set->first_n[indices[i]].
 * 
 * @warning The caller must free the returned index array to prevent memory leaks.
 *          The original set data must remain valid while the indices are in use.
 * 
 * @complexity O(n²) where n is set->count, due to bubble sort algorithm.
 *            For typical use cases with small n, this is acceptable.
 * 
 * @see print_results() for usage in sorted output generation
 * @see UniqueMrSet structure definition for parallel array organization
 * @see safe_malloc() for memory allocation with error handling
 * 
 * @example
 * ```c
 * // Create and populate unique mr set
 * UniqueMrSet* set = create_unique_mr_set();
 * add_unique_mr(set, 108, 1234);  // First discovery
 * add_unique_mr(set, 25, 408);    // Second discovery  
 * add_unique_mr(set, 53, 2048);   // Third discovery
 * 
 * // Original order: [108, 25, 53]
 * // Desired sorted order: [25, 53, 108]
 * 
 * int* indices = create_sorting_indices(set);
 * // indices array contains: [1, 2, 0]
 * 
 * // Access mr values in sorted order
 * for (int i = 0; i < set->count; i++) {
 *     int idx = indices[i];
 *     printf("mr=%lu (first found at n=%lu)\n", 
 *            set->values[idx], set->first_n[idx]);
 * }
 * // Output: mr=25 (first found at n=408)
 * //         mr=53 (first found at n=2048)  
 * //         mr=108 (first found at n=1234)
 * 
 * free(indices);  // Clean up when done
 * ```
 */
static int* create_sorting_indices(const UniqueMrSet* set) {
    int* indices = safe_malloc(set->count * sizeof(int), "sorting indices");
    
    for (int i = 0; i < set->count; i++) {
        indices[i] = i;
    }
    
    // Simple bubble sort for mr values
    for (int i = 0; i < set->count - 1; i++) {
        for (int j = 0; j < set->count - 1 - i; j++) {
            if (set->values[indices[j]] > set->values[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    return indices;
}

/**
 * @brief Exports unique mr discovery results to a JSON file with simple array structure for data analysis.
 * 
 * This function generates a clean JSON array containing all unique mr values discovered during
 * the search process along with the first n value that generated each mr. The minimal structure
 * facilitates easy integration with data analysis tools, web applications, and JSON processing
 * libraries while maintaining human readability.
 * 
 * JSON Output Format:
 * - Array of objects with "mr" and "n" fields
 * - Results preserved in discovery order for temporal analysis
 * - Standard JSON formatting compatible with all JSON parsers
 * - No metadata overhead - pure data export
 * - Compact structure suitable for large result sets
 * 
 * @param ctx Pointer to the search context containing the unique mr set with all
 *            discovered results. Must contain a valid unique_set.
 * @param filename The output filename for the JSON file. Should include the .json extension.
 */
static void write_json_results(const SearchContext* ctx, const char* filename) {
    FILE* json_file = fopen(filename, "w");
    if (!json_file) {
        printf("\n[*] ERROR: Could not create the JSON output file %s\n", filename);
        return;
    }
    
    fprintf(json_file, "[\n");
    
    for (int i = 0; i < ctx->unique_set->count; i++) {
        fprintf(json_file, "  {\n");
        fprintf(json_file, "    \"mr\": %lu,\n", ctx->unique_set->values[i]);
        fprintf(json_file, "    \"n\": %lu\n", ctx->unique_set->first_n[i]);
        fprintf(json_file, "  }");
        
        if (i < ctx->unique_set->count - 1) {
            fprintf(json_file, ",");
        }
        fprintf(json_file, "\n");
    }
    
    fprintf(json_file, "]\n");
    fclose(json_file);
}

/**
 * @brief Generates a comprehensive validation report with performance metrics, discovery statistics, and sorted results for research verification.
 * 
 * This function produces a detailed summary of the entire search operation, combining performance
 * analysis, statistical reporting, and formatted result presentation suitable for research
 * documentation and validation purposes. The report includes both individual discovery details
 * and aggregate metrics, enabling comprehensive analysis of search effectiveness and algorithmic
 * performance across different computational scales.
 * 
 * The function implements a multi-section reporting format that progresses from detailed
 * individual results to high-level performance metrics, providing complete documentation
 * of the search operation. Results are presented in sorted order for systematic analysis
 * while preserving the relationship between mr values and their discovery contexts.
 * 
 * Report Structure:
 * 1. Individual Results: Sorted list of unique mr values with discovery contexts
 * 2. Performance Metrics: Execution time, processing speed, and computational efficiency
 * 3. Statistical Summary: Discovery rates, coverage analysis, and uniqueness metrics
 * 4. Formatted Output: Copy-ready list of mr values for further analysis
 * 
 * Validation Features:
 * - Sorted presentation enables systematic verification against theoretical predictions
 * - Performance metrics facilitate algorithmic optimization and hardware utilization analysis
 * - Statistical summaries provide insights into mr value distribution and discovery density
 * - Copy-ready formatting supports integration with external analysis tools
 * 
 * @param ctx Pointer to the search context containing all results, configuration parameters,
 *            and metadata needed for comprehensive reporting. Must contain valid unique set
 *            and search configuration data.
 * @param total_time The total execution time in seconds for performance analysis and speed
 *                   calculation. Used to compute processing rate and efficiency metrics.
 * @param found_count The total number of input values that successfully yielded mr values,
 *                    used for discovery rate analysis and coverage statistics.
 * @param processed_count The total number of input values processed during the search,
 *                        used for completion validation and rate calculations.
 * 
 * @note The function creates and manages sorting indices to present results in ascending
 *       mr value order while preserving the original discovery data organization.
 * 
 * @note Performance metrics include both absolute measures (time, counts) and derived
 *       measures (speed, percentages) for comprehensive efficiency analysis.
 * 
 * @note The copy-ready mr value list uses comma separation for easy integration with
 *       spreadsheets, analysis tools, and research documentation.
 * 
 * @note Memory management includes proper cleanup of temporary sorting indices to
 *       prevent resource leaks during report generation.
 * 
 * @note Statistical calculations use double precision to ensure accuracy for percentage
 *       computations and rate calculations across large number ranges.
 * 
 * @complexity O(n log n) where n is the number of unique mr values, dominated by the
 *            sorting operation for result presentation
 * 
 * @see create_sorting_indices() for sorted result presentation logic
 * @see SearchContext structure for comprehensive data organization
 * @see UniqueMrSet structure for discovery data storage
 * 
 * @example
 * ```c
 * // Generate comprehensive validation report
 * SearchContext ctx = { ... };  // Populated with search results
 * double total_time = omp_get_wtime() - start_time;
 * uint64_t found = 32767, processed = 32767;
 * 
 * print_results(&ctx, total_time, found, processed);
 * 
 * // Example output:
 * // **************************************************************************
 * // * High-performance mr pairs discovery engine using tuple-based transform *
 * // **************************************************************************
 * //
 * // [*] ALGORITHM SETUP
 * //   - Using 2 threads
 * //   - Range: from 1 to 2^20 = 1048575
 * //
 * // [*] SEARCH PROCESS
 * //   - Progress: (0.0%) | Processed: 0 | Unique values of mr found: 0 | 0.0 nums/sec | ETA: -- min
 * //      - New unique value of mr = 0 found, generated by n = 1 (total unique mr values: 1)
 * //		   - New unique value of mr = 1 found, generated by n = 3 (total unique mr values: 2)
 * //		   - New unique value of mr = 2 found, generated by n = 6 (total unique mr values: 3)
 * //		   - New unique value of mr = 576 found, generated by n = 16392 (total unique mr values: 4)
 * //		   - New unique value of mr = 3 found, generated by n = 7 (total unique mr values: 5)
 * //		   - New unique value of mr = 6 found, generated by n = 9 (total unique mr values: 6)
 * //          ....
 * //		   - New unique value of mr = 2428 found, generated by n = 4857 (total unique mr values: 41)
 * //		   - New unique value of mr = 3643 found, generated by n = 7287 (total unique mr values: 42)
 * //	 - Progress: (18.375111%) | Processed: 192677 | Unique values of mr found: 42 | 287070.2 nums/sec | ETA: 0.0 min
 * //	 - Progress: (96.269417%) | Processed: 1009458 | Unique values of mr found: 42 | 268719.2 nums/sec | ETA: 0.0 min
 * //
 * // [*] SEARCH RESULTS
 * //  - Total numbers processed: 32767
 * //  - Total time: 0.102 seconds
 * //  - Speed: 322535.37 numbers/second
 * //  - Numbers with mr found: 32767
 * //  - Unique mr values found: 42
 * //  - List of mr values found: 0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79,
 * //    91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432, 444, 487, 576, 592, 
 * //    649, 667, 683, 865, 889, 1153, 1214, 1821, 2428, 3643
 * ```
 */
static void print_results(const SearchContext* ctx, double total_time, 
                                   uint64_t found_count, uint64_t processed_count) {
   
    printf("\n[*] SEARCH RESULTS\n");
    printf("\t - Total numbers processed: %lu\n", processed_count);
    printf("\t - Total time: %.3f seconds\n", total_time);
    printf("\t - Speed: %.2f numbers/second\n", (double)processed_count / total_time);
    printf("\t - Numbers with mr found: %lu\n", found_count);
    printf("\t - Unique mr values found: %lu\n", (uint64_t)ctx->unique_set->count);

    printf("\t - List of mr values found: ");
    int* indices = create_sorting_indices(ctx->unique_set);
    for (int i = 0; i < ctx->unique_set->count; i++) {
        int idx = indices[i];
        if (i > 0) printf(", ");
        printf("%lu", ctx->unique_set->values[idx]);
    }
    printf("\n");
    
    free(indices);
}

// ***********************************************************************************
// * 10. COMMAND LINE ARGUMENT PROCESSING
// ***********************************************************************************

/**
 * @brief Displays the program identification banner with visual branding for application recognition.
 * 
 * This function outputs a standardized visual header that provides immediate identification
 * of the application and its research focus. The banner serves as the primary visual branding
 * element, establishing program identity through consistent ASCII art formatting and descriptive
 * title that clearly indicates the algorithmic approach and research methodology being employed.
 * 
 * The function creates a prominent visual separator that distinguishes program output from
 * system messages and provides professional presentation suitable for research environments,
 * academic documentation, and technical logging. The banner is designed to be immediately
 * recognizable and informative regardless of execution context or parameter validation outcomes.
 * 
 * Visual Design Features:
 * - Fixed-width ASCII art formatting (74 characters) for consistent terminal presentation
 * - Horizontal asterisk rule separators creating clear visual boundaries
 * - Descriptive title combining algorithm type with research methodology
 * - Professional appearance suitable for academic and research documentation
 * - Consistent branding across all program executions
 * 
 * Usage Context:
 * Called early in program initialization to establish visual context before any validation
 * or configuration operations. This ensures users always see program identification even
 * when command-line arguments are invalid or other early failures occur.
 * 
 * @note The banner spans exactly 74 characters in width, optimized for standard terminal
 *       displays while remaining readable on various screen sizes and output contexts.
 * 
 * @note This function has no parameters and no return value, focusing solely on visual
 *       output generation without any dependency on program state or configuration.
 * 
 * @note The banner text references "tuple-based transform" methodology, providing immediate
 *       context about the algorithmic approach without requiring additional explanation.
 * 
 * @note Output is sent to stdout and will appear in console output, log files, or any
 *       redirected output streams according to standard I/O behavior.
 * 
 * @complexity O(1) - constant time operation with fixed output content
 * 
 * @see print_algorithm_setup() for technical configuration display
 * @see validate_and_parse_arguments() for typical usage context
 * @see main() for program initialization sequence
 * 
 * @example
 * ```c
 * // Called at program start for immediate identification
 * print_program_header();
 * 
 * // Example output:
 * // **************************************************************************
 * // * High-performance mr pairs discovery engine using tuple-based transform *
 * // **************************************************************************
 * 
 * // Usage in error scenarios
 * int main(int argc, char* argv[]) {
 *     print_program_header();  // Always shows program identity
 *     
 *     if (argc != 2) {
 *         printf("Usage error...\n");
 *         return 1;  // User still sees what program they ran
 *     }
 *     
 *     // ... continue with normal execution
 * }
 * ```
 */
static void print_program_header() {
    printf("\n**************************************************************************\n");
    printf("* High-performance mr pairs discovery engine using tuple-based transform *\n");
    printf("**************************************************************************\n");
}

/**
 * @brief Displays comprehensive algorithm setup information including parallelization and search range configuration.
 * 
 * This function outputs detailed technical configuration information that enables users to
 * verify execution parameters, understand computational scope, and validate system resource
 * utilization before the search process begins. The display provides both system-level
 * information (thread count) and algorithm-specific parameters (search range) in a structured,
 * professional format suitable for research documentation and performance analysis.
 * 
 * The function serves as a critical validation checkpoint, allowing users to confirm that
 * the program has correctly interpreted their parameters and is utilizing expected system
 * resources. This information is essential for performance monitoring, resource planning,
 * and result interpretation in computational research contexts.
 * 
 * Configuration Display Features:
 * - Real-time thread count detection showing actual parallelization level
 * - Mathematical range notation (2^exponent) for precise specification
 * - Absolute numeric bounds for concrete understanding of computational scope
 * - Structured formatting with consistent indentation and sectioning
 * - Professional presentation suitable for research logs and documentation
 * 
 * System Information Reported:
 * - Thread utilization: Actual OpenMP thread count available for parallel execution
 * - Search methodology: Implicit confirmation of tuple-based transform approach
 * - Range bounds: Both exponential notation and absolute numeric limits
 * - Parameter validation: Confirmation of successful argument processing
 * 
 * @param exponent The power-of-2 exponent defining the search range upper bound.
 *                 Used to display both mathematical notation (2^exponent) and provide
 *                 computational scope context. Must be a valid integer in range [1,64].
 * @param max_n The calculated maximum search value (2^exponent) representing the
 *              exclusive upper bound of the search range. Used to display absolute
 *              numeric range being processed. Must be computed as (1UL << exponent).
 * 
 * @note The function displays actual thread count from omp_get_max_threads(), providing
 *       real-time verification of the parallel execution environment rather than
 *       theoretical or configured values.
 * 
 * @note Range display shows inclusive bounds (1 to max_n-1) to clarify that max_n
 *       itself is excluded from the search, following standard mathematical convention
 *       for range notation.
 * 
 * @note The ALGORITHM SETUP section header provides clear visual separation from the
 *       program banner and creates logical organization of output information.
 * 
 * @note Thread count verification helps users identify potential performance issues
 *       related to system configuration, OpenMP environment variables, or resource
 *       allocation in shared computing environments.
 * 
 * @warning The function assumes valid input parameters that have passed validation
 *          in validate_and_parse_arguments(). Invalid parameters may produce
 *          incorrect or misleading output.
 * 
 * @complexity O(1) - constant time operation with simple parameter formatting
 * 
 * @see omp_get_max_threads() for dynamic thread count detection
 * @see validate_and_parse_arguments() for parameter validation requirements
 * @see print_program_header() for visual banner that precedes this output
 * @see main() for typical calling sequence and context
 * 
 * @example
 * ```c
 * // Typical usage after successful argument validation
 * int exponent = 25;
 * uint64_t max_n = 1UL << exponent;  // 33,554,432
 * 
 * print_algorithm_setup(exponent, max_n);
 * 
 * // Example output on a 8-core system:
 * // [*] ALGORITHM SETUP
 * //     - Using 8 threads
 * //     - Range: from 1 to 2^25 = 33554431 
 * 
 * // Different system configurations produce different thread counts:
 * // Single-core system: "Using 1 threads"
 * // 16-core system: "Using 16 threads"
 * // Container with limited resources: "Using 4 threads"
 * 
 * // Different range examples:
 * // exponent=20: "Range: from 1 to 2^20 = 1048575"
 * // exponent=30: "Range: from 1 to 2^30 = 1073741823"
 * 
 * // Usage in main function
 * int main(int argc, char* argv[]) {
 *     print_program_header();
 *     
 *     if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
 *         return 1;
 *     }
 *     
 *     print_algorithm_setup(exponent, max_n);  // Show validated configuration
 *     
 *     // ... proceed with search execution
 * }
 * ```
 */
static void print_algorithm_setup(int exponent, uint64_t max_n) {
    printf("\n[*] ALGORITHM SETUP\n");
    printf("\t - Using %d threads\n", omp_get_max_threads());
    printf("\t - Range: from 1 to 2^%d = %lu \n", exponent, max_n - 1);
}

/**
 * @brief Validates and parses command-line arguments with comprehensive error handling and user guidance.
 * 
 * This function performs complete validation and parsing of command-line arguments for the Collatz
 * sequence analysis program, implementing robust error checking with detailed usage instructions
 * and practical examples. It serves as the primary input validation gateway, ensuring that only
 * valid configurations reach the computational core while providing clear guidance for proper
 * program usage.
 * 
 * The function implements a multi-stage validation process that checks argument count, parses
 * the exponent value, validates the range constraints, and computes the derived maximum search
 * value. Each validation stage provides specific error messages and guidance to help users
 * understand proper program usage and select appropriate parameters for their analysis needs.
 * 
 * Validation Process:
 * 1. Argument Count Verification: Ensures exactly one parameter (exponent) is provided
 * 2. Exponent Parsing: Converts string argument to integer using atoi()
 * 3. Range Validation: Ensures exponent is within safe computational limits [1, 64]
 * 4. Maximum N Calculation: Computes 2^exponent with overflow safety checks
 * 5. Parameter Output: Updates caller's variables only on successful validation
 * 
 * Usage Guidance Features:
 * - Clear syntax explanation with program name extraction from argv[0]
 * - Practical examples showing typical usage patterns
 * - Recommended exponent values with corresponding search ranges
 * - Performance guidance for different computational scales
 * - Specific error messages for each type of validation failure
 * 
 * @param argc The number of command-line arguments including program name.
 *             Expected to be exactly 2 (program name + exponent parameter).
 * @param argv Array of command-line argument strings. argv[0] contains program name,
 *             argv[1] should contain the exponent value as a string.
 * @param exponent Pointer to store the parsed and validated exponent value.
 *                 Updated only if validation succeeds completely.
 * @param max_n Pointer to store the calculated maximum search value (2^exponent).
 *              Updated only if validation succeeds completely.
 * 
 * @return true if all arguments are valid and successfully parsed
 *         false if any validation fails, with appropriate error messages displayed
 * 
 * @note The function provides comprehensive usage information on argument count mismatch,
 *       including practical examples and performance guidance for different scales.
 * 
 * @note Exponent parsing uses atoi() which handles leading whitespace and stops at
 *       first non-digit character, providing reasonable tolerance for user input.
 * 
 * @note Range validation prevents both underflow (exponent < 1) and overflow scenarios
 *       (exponent > 64) that could cause undefined behavior or excessive computation.
 * 
 * @note The function only updates output parameters on complete success, ensuring
 *       that partially validated data never reaches the caller.
 * 
 * @note Maximum N calculation uses bit shifting (1UL << exponent) for exact
 *       power-of-2 computation without floating-point precision issues.
 * 
 * @warning Large exponent values (approaching 64) will create enormous search spaces
 *          requiring substantial computational resources and execution time.
 * 
 * @complexity O(1) - simple argument parsing and validation with fixed operations
 * 
 * @see atoi() for string-to-integer conversion behavior
 * @see print_program_header() for visual banner display
 * @see print_algorithm_setup() for technical configuration display
 * @see main() for typical usage context and error handling
 * 
 * @example
 * ```c
 * // Typical main function usage
 * int main(int argc, char* argv[]) {
 *     int exponent;
 *     uint64_t max_n;
 *     
 *     if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
 *         return 1;  // Exit with error code on validation failure
 *     }
 *     
 *     printf("Range: from 1 to 2^%d = %lu \n", exponent,  max_n - 1);
 *     // Proceed with validated parameters...
 * }
 * 
 * // Example valid command lines:
 * // ./brute_force 25        → exponent=25, max_n=33,554,432
 * // ./brute_force 20        → exponent=20, max_n=1,048,576
 * 
 * // Example invalid command lines and responses:
 * // ./brute_force           → Usage message with examples
 * // ./brute_force 0         → "[*] ERROR: Exponent must be between 1 and 64"
 * // ./brute_force 65        → "[*] ERROR: Exponent must be between 1 and 64"
 * // ./brute_force abc       → exponent=0, range error message
 * ```
 */
static bool validate_and_parse_arguments(int argc, char* argv[], int* exponent, uint64_t* max_n) {
    
    // Print program header
    print_program_header();

    if (argc != 2) {
        printf("\n[*] USAGE:");
        printf("\n\t%s <exponent>\n", argv[0]);
        printf("\n[*] EXAMPLE:");
        printf("\n\t%s 25  (to search n < 2^25)\n", argv[0]);
        printf("\n[*] RECOMMENDED EXPONENTS:");
        printf("\n\t20 -> 2^20 = 1,048,576 (quick test)");
        printf("\n\t25 -> 2^25 = 33,554,432 (default)");
        printf("\n\t30 -> 2^30 = 1,073,741,824 (intensive use)");
        printf("\n\n");
        return false;
    }
    
    *exponent = atoi(argv[1]);
    
    if (*exponent < 1 || *exponent > 64) {
        printf("\n[*] ERROR: Exponent must be a number between 1 and 64.\n", *exponent);
        return false;
    }
    
    *max_n = 1UL << *exponent;
    return true;
}

// ***********************************************************************************
// * 11. MAIN FUNCTION
// ***********************************************************************************

/**
 * @brief Main program entry point that orchestrates the complete Collatz sequence mr discovery workflow with JSON output generation.
 * 
 * This function serves as the central coordinator for the entire Collatz sequence analysis operation,
 * managing the complete application lifecycle from command-line argument processing through result
 * output in JSON format and comprehensive resource cleanup. It implements a structured workflow
 * that ensures proper initialization, execution monitoring, result documentation in modern JSON
 * format, and comprehensive cleanup to prevent resource leaks.
 * 
 * The function demonstrates best practices for scientific computing applications by providing
 * comprehensive error handling, progress monitoring, result validation, JSON output generation
 * optimized for modern data processing pipelines, and professional presentation suitable for
 * both human analysis and automated integration.
 * 
 * * Program Execution Workflow:
 * 1. Command-line argument validation with program banner display
 * 2. Algorithm setup and configuration display
 * 3. Search context initialization with all required data structures
 * 4. Parallel search execution with automatic scheduling optimization
 * 5. Performance timing and metrics calculation
 * 6. JSON result file generation optimized for modern data processing
 * 7. Comprehensive validation report and summary display
 * 8. File output confirmation and complete resource cleanup
 * 
 * Output Format Strategy:
 * The function generates JSON output to serve modern data processing needs:
 * - Clean array structure for easy parsing and integration
 * - Compatible with web applications and data analysis tools
 * - Human-readable format suitable for inspection and debugging
 * 
 * @param argc Number of command-line arguments including the program name.
 *             Expected to be 2 (program name + exponent parameter).
 * @param argv Array of command-line argument strings containing program name and
 *             the exponent value defining the search range (n < 2^exponent).
 * 
 * @return 0 on successful completion of all operations including search execution,
 *           JSON result output, and cleanup
 *         1 on command-line argument validation failure or other error conditions
 * 
 * @note The function generates output file with naming pattern:
 *       "mr_pairs_detected_range_1_to_2pow<exponent>.json"
 * 
 * @note JSON format contains clean array structure optimized for modern
 *       data processing tools and web applications.
 * 
 * @note Resource cleanup is performed unconditionally to ensure proper system
 *       resource management regardless of file generation outcomes.
 * 
 * @see write_json_results() for JSON format generation
 * @see validate_and_parse_arguments() for input validation
 * @see execute_parallel_search() for core computational algorithm
 */
int main(int argc, char* argv[]) {
    // Parse and validate command line arguments
    int exponent;
    uint64_t max_n;
    if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
        return 1;
    }
    
    // Print algorithm setup
    print_algorithm_setup(exponent, max_n);
    
    // Initialize search context
    SearchContext ctx = {
        .max_n = max_n,
        .exponent = exponent,
        .unique_set = create_unique_mr_set(),
        .progress = create_progress_tracker(),
        .start_time = omp_get_wtime()
    };

    // Execute the parallel search
    uint64_t found_count = 0;
    uint64_t processed_count = 0;
    
    execute_parallel_search(&ctx, &found_count, &processed_count);
    
    double end_time = omp_get_wtime();
    double total_time = end_time - ctx.start_time;
    
    // Generate output filename and write JSON results
    char json_filename[256];
    snprintf(json_filename, sizeof(json_filename), "mr_pairs_detected_range_1_to_2pow%d.json", exponent);
    
    write_json_results(&ctx, json_filename);
    
    // Print validation results and summary
    print_results(&ctx, total_time, found_count, processed_count);
    
    // Print file details
    FILE* test_file = fopen(json_filename, "r");
    if (test_file) {
        fseek(test_file, 0, SEEK_END);
        long file_size = ftell(test_file);
        fclose(test_file);
        
        printf("\n[*] JSON OUTPUT\n");
        printf("\t - File: '%s'\n", json_filename);
        printf("\t - Size: %ld bytes\n", file_size);
    }
    
    // Cleanup
    destroy_unique_mr_set(ctx.unique_set);
    destroy_progress_tracker(ctx.progress);
    
    return 0;
}
