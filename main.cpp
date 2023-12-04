#include <vector>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cassert>
#include <iostream>
#include <cmath>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>

static constexpr bool DEBUG_PRINT_VEC = false;

static constexpr int32_t MAX_NUM_PROCESSES = 4;
static constexpr u_int64_t VEC_SIZE = std::pow(10, 8);
static constexpr size_t TESTS = 5;
static constexpr u_int32_t BLOCK = 500;


using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline int64_t get_microseconds(const time_point &start, const time_point &end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline int64_t get_milliseconds(const time_point &start, const time_point &end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

inline int64_t get_seconds(const time_point &start, const time_point &end)
{
    return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
}


template<typename T>
void generate_random_vec(T &data)
{
    std::srand(unsigned(std::time(nullptr)));
    std::generate(std::begin(data), std::end(data), std::rand);
}

template<typename T>
void generate_data(T &seq_data, T &par_data)
{
    seq_data.resize(TESTS);
    for (size_t i = 0; i < TESTS; ++i)
    {
        seq_data[i].resize(VEC_SIZE);
        generate_random_vec(seq_data[i]);    
    }

    par_data = seq_data;
}

template<typename T>
void print_vec(std::vector<T> &vec)
{
    for (const auto &el: vec) {
        std::cout << el << ' ';
    }
    std::cout << '\n';
}

template<typename iter, typename compare>
iter get_partition(iter first, iter second, compare cmp)
{
    auto pivot = std::prev(second, 1);
    auto i = first;
    for (auto j = first; j != pivot; ++j){
        // bool format 
        if (cmp(*j, *pivot)){
            std::swap(*i++, *j);
        }
    }
    std::swap(*i, *pivot);
    return i;
}

template<typename iter, typename compare>
void sequential_quick_sort(iter first, iter second, compare cmp)
{
    if (std::distance(first, second) > 1)
    {
        iter bound = get_partition(first, second, cmp);
        sequential_quick_sort(first, bound, cmp);
        sequential_quick_sort(bound + 1, second, cmp);
    }
}

template<typename iter, typename compare>
void parallel_quick_sort(iter first, iter second, compare cmp)
{
    // std::cout << "Thread #" << oneapi::tbb::this_task_arena::current_thread_index() << " out of " << oneapi::tbb::this_task_arena::max_concurrency() << '\n';
    // assert(oneapi::tbb::this_task_arena::current_thread_index() < MAX_NUM_PROCESSES);
    if (std::distance(first, second) > BLOCK)
    {
        iter bound = get_partition(first, second, cmp);
        oneapi::tbb::parallel_invoke(   [=]{parallel_quick_sort(first,      bound,  cmp);},
                                        [=]{parallel_quick_sort(bound + 1,  second, cmp);});
    }
    else
    {
        sequential_quick_sort(first, second, cmp);
    }
}

// Практическое задание #1.

// Нужно реализовать quicksort. От Вас требуется написать последовательную версию алгоритма  (seq) и параллельную версию (par). 
// Взять случайный массив из 10^8 элементов и отсортировать. (Усреднить по 5 запускам) 
// Сравнить время работы par на 4 процессах и seq на одном процессе - у Вас должно быть раза в 3 быстрее.  (Если будет медленнее, то выставление баллов оставляется на моё усмотрение.)

// Дедлайн где-то в начале декабря.  Нужен код на гитхабе и результаты запусков в README.md. Код, который запускает, тоже должен лежать в репо.

int main() 
{
    std::vector<std::vector<int>> seq_vecs;
    std::vector<std::vector<int>> par_vecs;
    
    generate_data(seq_vecs, par_vecs);

    int64_t seq_total_time = 0;
    int64_t par_total_time = 0;

    // Sequential
    {
        for (size_t i = 0; i < TESTS; ++i)
        {
            auto &vec = seq_vecs[i];
            if (DEBUG_PRINT_VEC)
            {
                print_vec(vec);
            }

            {
                time_point start = std::chrono::high_resolution_clock::now();
                sequential_quick_sort(std::begin(vec), std::end(vec), std::less<int>());
                time_point end = std::chrono::high_resolution_clock::now();
                std::cout << "Sequential #" << i << " timer finished: "    << get_microseconds(start, end)   << " microseconds\t/\t" \
                                                    << get_milliseconds(start, end)   << " milliseconds\t/\t" \
                                                    << get_seconds(start, end)        << " seconds\n\n";
                seq_total_time += get_microseconds(start, end);
            }

            if (DEBUG_PRINT_VEC)
            {
                print_vec(vec);
            }
        }
        std::cout << "Sequential mean time: " << (float)seq_total_time / (float)TESTS << " microseconds\n\n";
    }

    // Parallel
    {
        // Limiting max number of processes
        oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, MAX_NUM_PROCESSES);
        
        for (size_t i = 0; i < TESTS; ++i)
        {
            auto &vec = par_vecs[i];
            if (DEBUG_PRINT_VEC)
            {
                print_vec(vec);
            }

            {
                time_point start = std::chrono::high_resolution_clock::now();
                parallel_quick_sort(std::begin(vec), std::end(vec), std::less<int>());
                time_point end = std::chrono::high_resolution_clock::now();
                std::cout << "Parallel #" << i << " timer finished: "    << get_microseconds(start, end)   << " microseconds\t/\t" \
                                                    << get_milliseconds(start, end)   << " milliseconds\t/\t" \
                                                    << get_seconds(start, end)        << " seconds\n\n";
                par_total_time += get_microseconds(start, end);
            }

            if (DEBUG_PRINT_VEC)
            {
                print_vec(vec);
            }
        }
        std::cout << "Parallel mean time: " << (float)par_total_time / (float)TESTS << " microseconds\n\n";
    }

    std::cout << "Sequential to Parallel Ratio: " << (float)seq_total_time / (float)par_total_time << '\n';

    return 0;
}