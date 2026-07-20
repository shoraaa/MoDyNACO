#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

constexpr double kEps = 1e-10;

std::vector<double> read_square_matrix(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
    auto buf = array.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
        throw std::invalid_argument("expected a square matrix");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0] * buf.shape[1]);
}

std::vector<double> read_optional_square_matrix(const py::object& obj, std::size_t expected_dim) {
    if (obj.is_none()) {
        return {};
    }
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>(obj);
    auto buf = array.request();
    if (buf.ndim != 2 || static_cast<std::size_t>(buf.shape[0]) != expected_dim || static_cast<std::size_t>(buf.shape[1]) != expected_dim) {
        throw std::invalid_argument("optional matrix has incorrect shape");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + expected_dim * expected_dim);
}

std::vector<double> read_cube(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
    std::size_t expected_dim1,
    std::size_t expected_dim2) {
    auto buf = array.request();
    if (buf.ndim != 3 || static_cast<std::size_t>(buf.shape[1]) != expected_dim1 || static_cast<std::size_t>(buf.shape[2]) != expected_dim2) {
        throw std::invalid_argument("expected a rank-3 tensor with matching trailing dimensions");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0] * buf.shape[1] * buf.shape[2]);
}

std::vector<double> read_vector(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
    auto buf = array.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("expected a rank-1 vector");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

std::vector<double> read_matrix(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
    std::size_t expected_cols) {
    auto buf = array.request();
    if (buf.ndim != 2 || static_cast<std::size_t>(buf.shape[1]) != expected_cols) {
        throw std::invalid_argument("expected a rank-2 matrix with matching column count");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0] * buf.shape[1]);
}

}  // namespace

struct BatchTrace {
    std::vector<std::int64_t> paths_flat;
    std::int64_t problem_size = 0;
    std::int64_t n_ants = 0;

    py::array_t<std::int64_t> paths_numpy() const {
        py::array_t<std::int64_t> out({problem_size, n_ants});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < problem_size; ++row) {
            for (std::int64_t col = 0; col < n_ants; ++col) {
                buf(row, col) = paths_flat[static_cast<std::size_t>(row * n_ants + col)];
            }
        }
        return out;
    }
};

struct MultiBatchTrace {
    std::vector<std::int64_t> paths_flat;
    std::int64_t n_colonies = 0;
    std::int64_t problem_size = 0;
    std::int64_t n_ants = 0;

    py::array_t<std::int64_t> paths_numpy() const {
        py::array_t<std::int64_t> out({n_colonies, problem_size, n_ants});
        auto buf = out.mutable_unchecked<3>();
        for (std::int64_t colony = 0; colony < n_colonies; ++colony) {
            for (std::int64_t row = 0; row < problem_size; ++row) {
                for (std::int64_t col = 0; col < n_ants; ++col) {
                    const auto idx = static_cast<std::size_t>(((colony * problem_size) + row) * n_ants + col);
                    buf(colony, row, col) = paths_flat[idx];
                }
            }
        }
        return out;
    }
};

struct RouteTrace {
    std::vector<std::int64_t> paths_flat;
    std::int64_t seq_len = 0;
    std::int64_t n_ants = 0;

    py::array_t<std::int64_t> paths_numpy() const {
        py::array_t<std::int64_t> out({seq_len, n_ants});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < seq_len; ++row) {
            for (std::int64_t col = 0; col < n_ants; ++col) {
                buf(row, col) = paths_flat[static_cast<std::size_t>(row * n_ants + col)];
            }
        }
        return out;
    }
};

class ACO_TSP {
public:
    ACO_TSP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        int n_ants = 20,
        double decay = 0.9,
        double alpha = 1.0,
        double beta = 1.0,
        bool elitist = false,
        bool min_max = false,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        py::object min_obj = py::none(),
        std::uint64_t seed = std::random_device{}())
        : distances_(read_square_matrix(distances_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          problem_size_(static_cast<std::int64_t>(distances_array.shape(0))),
          rng_(seed) {
        if (n_ants_ <= 0) {
            throw std::invalid_argument("n_ants must be positive");
        }
        if (problem_size_ <= 1) {
            throw std::invalid_argument("problem size must be greater than 1");
        }

        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
            if (min_ <= 1e-9) {
                throw std::invalid_argument("min must be greater than 1e-9 when min_max is enabled");
            }
            max_ = -1.0;
        }

        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(problem_size_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(problem_size_ * problem_size_), 1.0);
            if (min_max_) {
                std::fill(pheromone_.begin(), pheromone_.end(), min_);
            }
        }

        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(problem_size_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(problem_size_ * problem_size_));
            for (std::int64_t row = 0; row < problem_size_; ++row) {
                for (std::int64_t col = 0; col < problem_size_; ++col) {
                    heuristic_[index(row, col)] = 1.0 / std::max(distances_[index(row, col)], kEps);
                }
            }
        }
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }

    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(pheromone_.size()))) != problem_size_) {
            throw std::invalid_argument("pheromone has incorrect shape");
        }
    }

    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        heuristic_ = read_square_matrix(heuristic_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(heuristic_.size()))) != problem_size_) {
            throw std::invalid_argument("heuristic has incorrect shape");
        }
    }

    py::array_t<double> get_pheromone() const { return matrix_to_numpy(pheromone_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy(heuristic_); }
    py::array_t<double> get_distances() const { return matrix_to_numpy(distances_); }

    std::vector<std::int64_t> get_shortest_path() const { return shortest_path_; }
    double get_lowest_cost() const { return lowest_cost_; }

    void sparsify(int k_sparse) {
        if (k_sparse <= 0 || k_sparse >= problem_size_) {
            throw std::invalid_argument("k_sparse must be in [1, problem_size - 1]");
        }

        std::vector<double> sparse(distances_.size(), 1e10);
        for (std::int64_t row = 0; row < problem_size_; ++row) {
            std::vector<std::pair<double, std::int64_t>> edges;
            edges.reserve(static_cast<std::size_t>(problem_size_ - 1));
            for (std::int64_t col = 0; col < problem_size_; ++col) {
                if (row == col) {
                    continue;
                }
                edges.emplace_back(distances_[index(row, col)], col);
            }
            std::partial_sort(
                edges.begin(),
                edges.begin() + k_sparse,
                edges.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            for (int i = 0; i < k_sparse; ++i) {
                sparse[index(row, edges[static_cast<std::size_t>(i)].second)] = edges[static_cast<std::size_t>(i)].first;
            }
        }

        heuristic_.resize(sparse.size());
        for (std::size_t i = 0; i < sparse.size(); ++i) {
            heuristic_[i] = 1.0 / std::max(sparse[i], kEps);
        }
    }

    py::tuple sample_trace() {
        auto sampled = sample_paths();
        py::array_t<double> costs = vector_to_numpy(sampled.costs);
        return py::make_tuple(costs, sampled.trace);
    }

    py::tuple sample_ant_heuristic_trace(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ant_heuristics_array,
        std::uint64_t seed = std::random_device{}()) const {
        auto ant_heuristics = read_cube(ant_heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        if (ant_heuristics_array.shape(0) != n_ants_) {
            throw std::invalid_argument("ant heuristics must have shape (n_ants, n, n)");
        }
        std::mt19937_64 local_rng(seed);
        auto sampled = sample_paths_with_ant_heuristics(pheromone_, ant_heuristics, local_rng);
        py::array_t<double> costs = vector_to_numpy(sampled.costs);
        return py::make_tuple(costs, sampled.trace);
    }

    py::tuple sample_many_traces(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristics_array,
        std::uint64_t seed = std::random_device{}()) const {
        auto heuristics = read_cube(heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        const auto n_colonies = static_cast<std::int64_t>(heuristics_array.shape(0));
        const auto matrix_size = static_cast<std::size_t>(problem_size_ * problem_size_);
        const auto path_block = static_cast<std::size_t>(problem_size_ * n_ants_);

        std::vector<double> costs_flat(static_cast<std::size_t>(n_colonies * n_ants_), 0.0);
        MultiBatchTrace trace;
        trace.n_colonies = n_colonies;
        trace.problem_size = problem_size_;
        trace.n_ants = n_ants_;
        trace.paths_flat.resize(static_cast<std::size_t>(n_colonies) * path_block);

        #pragma omp parallel for if (n_colonies > 1)
        for (std::int64_t colony = 0; colony < n_colonies; ++colony) {
            std::mt19937_64 local_rng(seed + static_cast<std::uint64_t>(colony));
            const auto heuristic_offset = static_cast<std::size_t>(colony) * matrix_size;
            std::vector<double> local_heuristic(
                heuristics.begin() + static_cast<std::ptrdiff_t>(heuristic_offset),
                heuristics.begin() + static_cast<std::ptrdiff_t>(heuristic_offset + matrix_size));
            auto sampled = sample_paths_with_matrices(pheromone_, local_heuristic, local_rng);

            const auto path_offset = static_cast<std::size_t>(colony) * path_block;
            std::copy(sampled.paths_flat.begin(), sampled.paths_flat.end(), trace.paths_flat.begin() + static_cast<std::ptrdiff_t>(path_offset));
            const auto cost_offset = static_cast<std::size_t>(colony) * static_cast<std::size_t>(n_ants_);
            std::copy(sampled.costs.begin(), sampled.costs.end(), costs_flat.begin() + static_cast<std::ptrdiff_t>(cost_offset));
        }

        py::array_t<double> costs({n_colonies, static_cast<std::int64_t>(n_ants_)});
        auto cost_buf = costs.mutable_unchecked<2>();
        for (std::int64_t colony = 0; colony < n_colonies; ++colony) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                cost_buf(colony, ant) = costs_flat[static_cast<std::size_t>(colony * n_ants_ + ant)];
            }
        }
        return py::make_tuple(costs, trace);
    }

    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        auto paths = read_paths(paths_array);
        return vector_to_numpy(gen_path_costs_vector(paths));
    }

    void update_pheromone_from_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        auto paths = read_paths(paths_array);
        auto costs = read_costs(costs_array);
        update_pheromone(paths, costs);
    }

    double run(int n_iterations) {
        if (n_iterations <= 0) {
            return lowest_cost_;
        }
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.costs);
        }
        return lowest_cost_;
    }

    py::array_t<double> run_many(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristics_array,
        int n_iterations,
        std::uint64_t seed = std::random_device{}()) const {
        if (n_iterations <= 0) {
            throw std::invalid_argument("n_iterations must be positive");
        }

        auto heuristics = read_cube(heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        const auto n_colonies = static_cast<std::int64_t>(heuristics_array.shape(0));
        const auto matrix_size = static_cast<std::size_t>(problem_size_ * problem_size_);
        std::vector<double> best_costs(static_cast<std::size_t>(n_colonies), std::numeric_limits<double>::infinity());

        #pragma omp parallel for if (n_colonies > 1)
        for (std::int64_t colony = 0; colony < n_colonies; ++colony) {
            std::mt19937_64 local_rng(seed + static_cast<std::uint64_t>(colony));
            const auto heuristic_offset = static_cast<std::size_t>(colony) * matrix_size;
            std::vector<double> local_heuristic(
                heuristics.begin() + static_cast<std::ptrdiff_t>(heuristic_offset),
                heuristics.begin() + static_cast<std::ptrdiff_t>(heuristic_offset + matrix_size));
            std::vector<double> local_pheromone = pheromone_;
            double local_best = std::numeric_limits<double>::infinity();
            std::vector<std::int64_t> local_shortest;
            double local_max = max_;

            for (int iter = 0; iter < n_iterations; ++iter) {
                auto sampled = sample_paths_with_matrices(local_pheromone, local_heuristic, local_rng);
                update_best_for_state(sampled.paths_flat, sampled.costs, local_pheromone, local_shortest, local_best, local_max);
                update_pheromone_for_state(sampled.paths_flat, sampled.costs, local_pheromone);
            }
            best_costs[static_cast<std::size_t>(colony)] = local_best;
        }

        return vector_to_numpy(best_costs);
    }

    double run_with_ant_heuristics(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ant_heuristics_array,
        int n_iterations,
        std::uint64_t seed = std::random_device{}()) const {
        if (n_iterations <= 0) {
            throw std::invalid_argument("n_iterations must be positive");
        }
        auto ant_heuristics = read_cube(ant_heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        if (ant_heuristics_array.shape(0) != n_ants_) {
            throw std::invalid_argument("ant heuristics must have shape (n_ants, n, n)");
        }

        std::mt19937_64 local_rng(seed);
        std::vector<double> local_pheromone = pheromone_;
        double local_best = std::numeric_limits<double>::infinity();
        std::vector<std::int64_t> local_shortest;
        double local_max = max_;

        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths_with_ant_heuristics(local_pheromone, ant_heuristics, local_rng);
            update_best_for_state(sampled.paths_flat, sampled.costs, local_pheromone, local_shortest, local_best, local_max);
            update_pheromone_for_state(sampled.paths_flat, sampled.costs, local_pheromone);
        }
        return local_best;
    }

private:
    struct SampledBatch {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        BatchTrace trace;
    };

    std::vector<double> distances_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    double min_ = 0.0;
    double max_ = -1.0;
    std::int64_t problem_size_;
    std::vector<std::int64_t> shortest_path_;
    double lowest_cost_ = std::numeric_limits<double>::infinity();
    std::mt19937_64 rng_;

    std::size_t index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * problem_size_ + col);
    }

    py::array_t<double> matrix_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out({problem_size_, problem_size_});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < problem_size_; ++row) {
            for (std::int64_t col = 0; col < problem_size_; ++col) {
                buf(row, col) = values[index(row, col)];
            }
        }
        return out;
    }

    py::array_t<double> vector_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out(values.size());
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = values[i];
        }
        return out;
    }

    std::vector<std::int64_t> read_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[0] != problem_size_ || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (problem_size, n_ants)");
        }
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + problem_size_ * n_ants_);
    }

    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) const {
        auto buf = costs_array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }

    SampledBatch sample_paths() {
        return sample_paths_with_matrices(pheromone_, heuristic_, rng_);
    }

    SampledBatch sample_paths_with_matrices(
        const std::vector<double>& pheromone,
        const std::vector<double>& heuristic,
        std::mt19937_64& rng) const {
        std::uniform_int_distribution<std::int64_t> start_dist(0, problem_size_ - 1);
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(problem_size_ * n_ants_));
        std::vector<unsigned char> valid(static_cast<std::size_t>(problem_size_ * n_ants_), 1);

        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto start = start_dist(rng);
            paths_flat[static_cast<std::size_t>(ant)] = start;
            valid[static_cast<std::size_t>(ant * problem_size_ + start)] = 0;
        }

        std::vector<double> weights(static_cast<std::size_t>(problem_size_));

        for (std::int64_t step = 1; step < problem_size_; ++step) {
            const std::int64_t prev_step = step - 1;
            for (int ant = 0; ant < n_ants_; ++ant) {
                const auto prev = paths_flat[static_cast<std::size_t>(prev_step * n_ants_ + ant)];
                double normalizer = 0.0;
                for (std::int64_t node = 0; node < problem_size_; ++node) {
                    const auto valid_idx = static_cast<std::size_t>(ant * problem_size_ + node);
                    if (!valid[valid_idx]) {
                        weights[static_cast<std::size_t>(node)] = 0.0;
                        continue;
                    }
                    const double pheromone_term = std::pow(std::max(pheromone[index(prev, node)], kEps), alpha_);
                    const double heuristic_term = std::pow(std::max(heuristic[index(prev, node)], kEps), beta_);
                    const double weight = pheromone_term * heuristic_term;
                    weights[static_cast<std::size_t>(node)] = weight;
                    normalizer += weight;
                }

                const double threshold = unit_dist(rng) * normalizer;
                double cumulative = 0.0;
                std::int64_t chosen = -1;
                std::int64_t last_feasible = -1;
                for (std::int64_t node = 0; node < problem_size_; ++node) {
                    if (weights[static_cast<std::size_t>(node)] > 0.0) {
                        last_feasible = node;
                    }
                    cumulative += weights[static_cast<std::size_t>(node)];
                    if (cumulative > threshold) {
                        chosen = node;
                        break;
                    }
                }
                if (chosen < 0) {
                    chosen = last_feasible >= 0 ? last_feasible : 0;
                }

                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = chosen;
                valid[static_cast<std::size_t>(ant * problem_size_ + chosen)] = 0;
            }
        }

        auto costs = gen_path_costs_vector(paths_flat);
        BatchTrace trace;
        trace.paths_flat = paths_flat;
        trace.problem_size = problem_size_;
        trace.n_ants = n_ants_;
        return SampledBatch{std::move(paths_flat), std::move(costs), std::move(trace)};
    }

    SampledBatch sample_paths_with_ant_heuristics(
        const std::vector<double>& pheromone,
        const std::vector<double>& ant_heuristics,
        std::mt19937_64& rng) const {
        std::uniform_int_distribution<std::int64_t> start_dist(0, problem_size_ - 1);
        std::vector<std::mt19937_64> ant_rngs(static_cast<std::size_t>(n_ants_));
        for (int ant = 0; ant < n_ants_; ++ant) {
            ant_rngs[static_cast<std::size_t>(ant)].seed(rng());
        }

        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(problem_size_ * n_ants_));
        std::vector<unsigned char> valid(static_cast<std::size_t>(problem_size_ * n_ants_), 1);

        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto start = start_dist(ant_rngs[static_cast<std::size_t>(ant)]);
            paths_flat[static_cast<std::size_t>(ant)] = start;
            valid[static_cast<std::size_t>(ant * problem_size_ + start)] = 0;
        }

        for (std::int64_t step = 1; step < problem_size_; ++step) {
            const std::int64_t prev_step = step - 1;
            #pragma omp parallel for if (n_ants_ > 1)
            for (int ant = 0; ant < n_ants_; ++ant) {
                std::vector<double> weights(static_cast<std::size_t>(problem_size_));
                std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
                const auto prev = paths_flat[static_cast<std::size_t>(prev_step * n_ants_ + ant)];
                double normalizer = 0.0;
                for (std::int64_t node = 0; node < problem_size_; ++node) {
                    const auto valid_idx = static_cast<std::size_t>(ant * problem_size_ + node);
                    if (!valid[valid_idx]) {
                        weights[static_cast<std::size_t>(node)] = 0.0;
                        continue;
                    }
                    const auto ant_offset = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_ * problem_size_);
                    const double pheromone_term = std::pow(std::max(pheromone[index(prev, node)], kEps), alpha_);
                    const double heuristic_term = std::pow(
                        std::max(ant_heuristics[ant_offset + index(prev, node)], kEps),
                        beta_);
                    const double weight = pheromone_term * heuristic_term;
                    weights[static_cast<std::size_t>(node)] = weight;
                    normalizer += weight;
                }

                const double threshold = unit_dist(ant_rngs[static_cast<std::size_t>(ant)]) * normalizer;
                double cumulative = 0.0;
                std::int64_t chosen = -1;
                std::int64_t last_feasible = -1;
                for (std::int64_t node = 0; node < problem_size_; ++node) {
                    if (weights[static_cast<std::size_t>(node)] > 0.0) {
                        last_feasible = node;
                    }
                    cumulative += weights[static_cast<std::size_t>(node)];
                    if (cumulative > threshold) {
                        chosen = node;
                        break;
                    }
                }
                if (chosen < 0) {
                    chosen = last_feasible >= 0 ? last_feasible : 0;
                }

                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = chosen;
                valid[static_cast<std::size_t>(ant * problem_size_ + chosen)] = 0;
            }
        }

        auto costs = gen_path_costs_vector(paths_flat);
        BatchTrace trace;
        trace.paths_flat = paths_flat;
        trace.problem_size = problem_size_;
        trace.n_ants = n_ants_;
        return SampledBatch{std::move(paths_flat), std::move(costs), std::move(trace)};
    }

    std::vector<double> gen_path_costs_vector(const std::vector<std::int64_t>& paths_flat) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double cost = 0.0;
            for (std::int64_t step = 0; step < problem_size_; ++step) {
                const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                const auto prev = paths_flat[static_cast<std::size_t>(((step == 0 ? problem_size_ - 1 : step - 1) * n_ants_) + ant)];
                cost += distances_[index(curr, prev)];
            }
            costs[static_cast<std::size_t>(ant)] = cost;
        }
        return costs;
    }

    void update_best(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        update_best_for_state(paths_flat, costs, pheromone_, shortest_path_, lowest_cost_, max_);
    }

    void update_best_for_state(
        const std::vector<std::int64_t>& paths_flat,
        const std::vector<double>& costs,
        std::vector<double>& pheromone,
        std::vector<std::int64_t>& shortest_path,
        double& lowest_cost,
        double& max_value) const {
        auto best_iter = std::min_element(costs.begin(), costs.end());
        if (best_iter == costs.end()) {
            return;
        }
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const double best_cost = *best_iter;
        if (best_cost >= lowest_cost) {
            return;
        }

        lowest_cost = best_cost;
        shortest_path.resize(static_cast<std::size_t>(problem_size_));
        for (std::int64_t step = 0; step < problem_size_; ++step) {
            shortest_path[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }

        if (min_max_) {
            const double next_max = static_cast<double>(problem_size_) / std::max(lowest_cost, kEps);
            if (max_value < 0.0) {
                const double pheromone_max = *std::max_element(pheromone.begin(), pheromone.end());
                if (pheromone_max > 0.0) {
                    const double scale = next_max / pheromone_max;
                    for (auto& value : pheromone) {
                        value *= scale;
                    }
                }
            }
            max_value = next_max;
        }
    }

    void update_pheromone(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        update_pheromone_for_state(paths_flat, costs, pheromone_);
    }

    void update_pheromone_for_state(
        const std::vector<std::int64_t>& paths_flat,
        const std::vector<double>& costs,
        std::vector<double>& pheromone) const {
        for (auto& value : pheromone) {
            value *= decay_;
        }

        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            const double deposit = 1.0 / std::max(*best_iter, kEps);
            deposit_path(paths_flat, best_idx, deposit, pheromone);
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                const double deposit = 1.0 / std::max(costs[static_cast<std::size_t>(ant)], kEps);
                deposit_path(paths_flat, ant, deposit, pheromone);
            }
        }

        if (min_max_) {
            for (auto& value : pheromone) {
                if (value > 1e-9 && value < min_) {
                    value = min_;
                }
                if (max_ > 0.0 && value > max_) {
                    value = max_;
                }
            }
        }
    }

    void deposit_path(const std::vector<std::int64_t>& paths_flat, int ant, double deposit, std::vector<double>& pheromone) const {
        for (std::int64_t step = 0; step < problem_size_; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto prev = paths_flat[static_cast<std::size_t>(((step == 0 ? problem_size_ - 1 : step - 1) * n_ants_) + ant)];
            pheromone[index(curr, prev)] += deposit;
            pheromone[index(prev, curr)] += deposit;
        }
    }
};

class ACO_CVRP {
public:
    ACO_CVRP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& demand_array,
        int n_ants = 20,
        double decay = 0.9,
        double alpha = 1.0,
        double beta = 1.0,
        bool elitist = false,
        bool min_max = false,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        py::object min_obj = py::none(),
        double capacity = 50.0,
        std::uint64_t seed = std::random_device{}())
        : distances_(read_square_matrix(distances_array)),
          demand_(read_vector(demand_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          capacity_(capacity),
          problem_size_(static_cast<std::int64_t>(distances_array.shape(0))),
          rng_(seed) {
        if (n_ants_ <= 0) {
            throw std::invalid_argument("n_ants must be positive");
        }
        if (problem_size_ <= 1) {
            throw std::invalid_argument("problem size must be greater than 1");
        }
        if (demand_.size() != static_cast<std::size_t>(problem_size_)) {
            throw std::invalid_argument("demand has incorrect shape");
        }

        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
            if (min_ <= 1e-9) {
                throw std::invalid_argument("min must be greater than 1e-9 when min_max is enabled");
            }
            max_ = -1.0;
        }

        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(problem_size_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(problem_size_ * problem_size_), 1.0);
            if (min_max_) {
                std::fill(pheromone_.begin(), pheromone_.end(), min_);
            }
        }

        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(problem_size_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(problem_size_ * problem_size_));
            for (std::int64_t row = 0; row < problem_size_; ++row) {
                for (std::int64_t col = 0; col < problem_size_; ++col) {
                    heuristic_[index(row, col)] = 1.0 / std::max(distances_[index(row, col)], kEps);
                }
            }
        }
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }

    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(pheromone_.size()))) != problem_size_) {
            throw std::invalid_argument("pheromone has incorrect shape");
        }
    }

    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        heuristic_ = read_square_matrix(heuristic_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(heuristic_.size()))) != problem_size_) {
            throw std::invalid_argument("heuristic has incorrect shape");
        }
    }

    py::array_t<double> get_pheromone() const { return matrix_to_numpy(pheromone_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy(heuristic_); }
    py::array_t<double> get_distances() const { return matrix_to_numpy(distances_); }
    py::array_t<double> get_demand() const { return vector_to_numpy(demand_); }
    std::vector<std::int64_t> get_shortest_path() const { return shortest_path_; }
    double get_lowest_cost() const { return lowest_cost_; }

    py::tuple sample_trace() {
        auto sampled = sample_paths();
        py::array_t<double> costs = vector_to_numpy(sampled.costs);
        return py::make_tuple(costs, sampled.trace);
    }

    py::tuple sample_ant_heuristic_trace(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ant_heuristics_array,
        std::uint64_t seed = std::random_device{}()) const {
        auto ant_heuristics = read_cube(ant_heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        if (ant_heuristics_array.shape(0) != n_ants_) {
            throw std::invalid_argument("ant heuristics must have shape (n_ants, n, n)");
        }
        std::mt19937_64 local_rng(seed);
        auto sampled = sample_paths_with_ant_heuristics(ant_heuristics, local_rng);
        py::array_t<double> costs = vector_to_numpy(sampled.costs);
        return py::make_tuple(costs, sampled.trace);
    }

    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        std::int64_t seq_len = 0;
        auto paths = read_paths(paths_array, seq_len);
        return vector_to_numpy(gen_path_costs_vector(paths, seq_len));
    }

    void update_pheromone_from_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        std::int64_t seq_len = 0;
        auto paths = read_paths(paths_array, seq_len);
        auto costs = read_costs(costs_array);
        update_pheromone(paths, seq_len, costs);
    }

    double run(int n_iterations) {
        if (n_iterations <= 0) {
            return lowest_cost_;
        }
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
        }
        return lowest_cost_;
    }

    double run_with_ant_heuristics(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& ant_heuristics_array,
        int n_iterations,
        std::uint64_t seed = std::random_device{}()) const {
        if (n_iterations <= 0) {
            throw std::invalid_argument("n_iterations must be positive");
        }
        auto ant_heuristics = read_cube(ant_heuristics_array, static_cast<std::size_t>(problem_size_), static_cast<std::size_t>(problem_size_));
        if (ant_heuristics_array.shape(0) != n_ants_) {
            throw std::invalid_argument("ant heuristics must have shape (n_ants, n, n)");
        }

        std::mt19937_64 local_rng(seed);
        ACO_CVRP local = *this;
        local.set_seed(seed);
        local.lowest_cost_ = std::numeric_limits<double>::infinity();
        local.shortest_path_.clear();
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = local.sample_paths_with_ant_heuristics(ant_heuristics, local_rng);
            local.update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
            local.update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
        }
        return local.lowest_cost_;
    }

private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        RouteTrace trace;
    };

    std::vector<double> distances_;
    std::vector<double> demand_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    double min_ = 0.0;
    double max_ = -1.0;
    double capacity_;
    std::int64_t problem_size_;
    std::vector<std::int64_t> shortest_path_;
    double lowest_cost_ = std::numeric_limits<double>::infinity();
    std::mt19937_64 rng_;

    std::size_t index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * problem_size_ + col);
    }

    py::array_t<double> matrix_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out({problem_size_, problem_size_});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < problem_size_; ++row) {
            for (std::int64_t col = 0; col < problem_size_; ++col) {
                buf(row, col) = values[index(row, col)];
            }
        }
        return out;
    }

    py::array_t<double> vector_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out(values.size());
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = values[i];
        }
        return out;
    }

    std::vector<std::int64_t> read_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        std::int64_t& seq_len) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (seq_len, n_ants)");
        }
        seq_len = buf.shape[0];
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + seq_len * n_ants_);
    }

    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) const {
        auto buf = costs_array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }

    void update_visit_mask(std::vector<unsigned char>& visit_mask, const std::vector<std::int64_t>& actions) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(actions[static_cast<std::size_t>(ant)])] = 0;
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_)] = 1;
            if (actions[static_cast<std::size_t>(ant)] == 0) {
                bool has_remaining = false;
                for (std::int64_t node = 1; node < problem_size_; ++node) {
                    if (visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] != 0) {
                        has_remaining = true;
                        break;
                    }
                }
                if (has_remaining) {
                    visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_)] = 0;
                }
            }
        }
    }

    void update_capacity_mask(
        const std::vector<std::int64_t>& actions,
        std::vector<double>& used_capacity,
        std::vector<unsigned char>& capacity_mask) const {
        std::fill(capacity_mask.begin(), capacity_mask.end(), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            if (actions[static_cast<std::size_t>(ant)] == 0) {
                used_capacity[static_cast<std::size_t>(ant)] = 0.0;
            }
            used_capacity[static_cast<std::size_t>(ant)] += demand_[static_cast<std::size_t>(actions[static_cast<std::size_t>(ant)])];
            const double remaining = capacity_ - used_capacity[static_cast<std::size_t>(ant)];
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                if (demand_[static_cast<std::size_t>(node)] > remaining) {
                    capacity_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] = 0;
                }
            }
        }
    }

    bool check_done(const std::vector<unsigned char>& visit_mask, const std::vector<std::int64_t>& actions) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            if (actions[static_cast<std::size_t>(ant)] != 0) {
                return false;
            }
            for (std::int64_t node = 1; node < problem_size_; ++node) {
                if (visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<std::int64_t> pick_move(
        const std::vector<std::int64_t>& prev,
        const std::vector<unsigned char>& visit_mask,
        const std::vector<unsigned char>& capacity_mask,
        std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        std::vector<double> weights(static_cast<std::size_t>(problem_size_));

        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto prev_node = prev[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node);
                if (visit_mask[idx] == 0 || capacity_mask[idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double pheromone_term = std::pow(std::max(pheromone_[index(prev_node, node)], kEps), alpha_);
                const double heuristic_term = std::pow(std::max(heuristic_[index(prev_node, node)], kEps), beta_);
                const double weight = pheromone_term * heuristic_term;
                weights[static_cast<std::size_t>(node)] = weight;
                normalizer += weight;
            }

            const double threshold = unit_dist(rng) * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                if (weights[static_cast<std::size_t>(node)] > 0.0) {
                    last_feasible = node;
                }
                cumulative += weights[static_cast<std::size_t>(node)];
                if (cumulative > threshold) {
                    chosen = node;
                    break;
                }
            }
            if (chosen < 0) {
                chosen = last_feasible >= 0 ? last_feasible : 0;
            }
            actions[static_cast<std::size_t>(ant)] = chosen;
        }
        return actions;
    }

    std::vector<std::int64_t> pick_move_ant_heuristics(
        const std::vector<std::int64_t>& prev,
        const std::vector<unsigned char>& visit_mask,
        const std::vector<unsigned char>& capacity_mask,
        const std::vector<double>& ant_heuristics,
        std::vector<std::mt19937_64>& ant_rngs) const {
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        #pragma omp parallel for if (n_ants_ > 1)
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
            std::vector<double> weights(static_cast<std::size_t>(problem_size_));
            const auto prev_node = prev[static_cast<std::size_t>(ant)];
            const auto ant_offset = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_ * problem_size_);
            double normalizer = 0.0;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node);
                if (visit_mask[idx] == 0 || capacity_mask[idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double pheromone_term = std::pow(std::max(pheromone_[index(prev_node, node)], kEps), alpha_);
                const double heuristic_term = std::pow(std::max(ant_heuristics[ant_offset + index(prev_node, node)], kEps), beta_);
                const double weight = pheromone_term * heuristic_term;
                weights[static_cast<std::size_t>(node)] = weight;
                normalizer += weight;
            }

            const double threshold = unit_dist(ant_rngs[static_cast<std::size_t>(ant)]) * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                if (weights[static_cast<std::size_t>(node)] > 0.0) {
                    last_feasible = node;
                }
                cumulative += weights[static_cast<std::size_t>(node)];
                if (cumulative > threshold) {
                    chosen = node;
                    break;
                }
            }
            if (chosen < 0) {
                chosen = last_feasible >= 0 ? last_feasible : 0;
            }
            actions[static_cast<std::size_t>(ant)] = chosen;
        }
        return actions;
    }

    SampledRoutes sample_paths() {
        std::vector<std::vector<std::int64_t>> path_steps;
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_visit_mask(visit_mask, actions);

        std::vector<double> used_capacity(static_cast<std::size_t>(n_ants_), 0.0);
        std::vector<unsigned char> capacity_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_capacity_mask(actions, used_capacity, capacity_mask);

        path_steps.push_back(actions);
        while (!check_done(visit_mask, actions)) {
            actions = pick_move(actions, visit_mask, capacity_mask, rng_);
            path_steps.push_back(actions);
            update_visit_mask(visit_mask, actions);
            update_capacity_mask(actions, used_capacity, capacity_mask);
        }

        const auto seq_len = static_cast<std::int64_t>(path_steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = path_steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }

        auto costs = gen_path_costs_vector(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(costs), std::move(trace)};
    }

    SampledRoutes sample_paths_with_ant_heuristics(
        const std::vector<double>& ant_heuristics,
        std::mt19937_64& rng) const {
        std::vector<std::mt19937_64> ant_rngs(static_cast<std::size_t>(n_ants_));
        for (int ant = 0; ant < n_ants_; ++ant) {
            ant_rngs[static_cast<std::size_t>(ant)].seed(rng());
        }

        std::vector<std::vector<std::int64_t>> path_steps;
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_visit_mask(visit_mask, actions);

        std::vector<double> used_capacity(static_cast<std::size_t>(n_ants_), 0.0);
        std::vector<unsigned char> capacity_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_capacity_mask(actions, used_capacity, capacity_mask);

        path_steps.push_back(actions);
        while (!check_done(visit_mask, actions)) {
            actions = pick_move_ant_heuristics(actions, visit_mask, capacity_mask, ant_heuristics, ant_rngs);
            path_steps.push_back(actions);
            update_visit_mask(visit_mask, actions);
            update_capacity_mask(actions, used_capacity, capacity_mask);
        }

        const auto seq_len = static_cast<std::int64_t>(path_steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = path_steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }

        auto costs = gen_path_costs_vector(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(costs), std::move(trace)};
    }

    std::vector<double> gen_path_costs_vector(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double cost = 0.0;
            for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
                const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
                cost += distances_[index(curr, next)];
            }
            costs[static_cast<std::size_t>(ant)] = cost;
        }
        return costs;
    }

    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        auto best_iter = std::min_element(costs.begin(), costs.end());
        if (best_iter == costs.end()) {
            return;
        }
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const double best_cost = *best_iter;
        if (best_cost >= lowest_cost_) {
            return;
        }

        lowest_cost_ = best_cost;
        shortest_path_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            shortest_path_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }

        if (min_max_) {
            const double next_max = static_cast<double>(problem_size_) / std::max(lowest_cost_, kEps);
            if (max_ < 0.0) {
                const double pheromone_max = *std::max_element(pheromone_.begin(), pheromone_.end());
                if (pheromone_max > 0.0) {
                    const double scale = next_max / pheromone_max;
                    for (auto& value : pheromone_) {
                        value *= scale;
                    }
                }
            }
            max_ = next_max;
        }
    }

    void update_pheromone(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }

        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            const double deposit = 1.0 / std::max(*best_iter, kEps);
            deposit_path(paths_flat, seq_len, best_idx, deposit);
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                const double deposit = 1.0 / std::max(costs[static_cast<std::size_t>(ant)], kEps);
                deposit_path(paths_flat, seq_len, ant, deposit);
            }
        }

        if (min_max_) {
            for (auto& value : pheromone_) {
                if (value > 1e-9 && value < min_) {
                    value = min_;
                }
                if (max_ > 0.0 && value > max_) {
                    value = max_;
                }
            }
        }

        for (auto& value : pheromone_) {
            if (value < 1e-10) {
                value = 1e-10;
            }
        }
    }

    void deposit_path(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

class ACO_VRPTW {
public:
    ACO_VRPTW(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& demand_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& windows_array,
        double service_time = 0.0,
        int n_ants = 20,
        double decay = 0.9,
        double alpha = 1.0,
        double beta = 1.0,
        bool elitist = false,
        bool min_max = false,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        py::object min_obj = py::none(),
        double capacity = 1.0,
        std::uint64_t seed = std::random_device{}())
        : distances_(read_square_matrix(distances_array)),
          demand_(read_vector(demand_array)),
          windows_(read_matrix(windows_array, 2)),
          service_time_(service_time),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          capacity_(capacity),
          problem_size_(static_cast<std::int64_t>(distances_array.shape(0))),
          rng_(seed) {
        if (n_ants_ <= 0) {
            throw std::invalid_argument("n_ants must be positive");
        }
        if (problem_size_ <= 1) {
            throw std::invalid_argument("problem size must be greater than 1");
        }
        if (demand_.size() != static_cast<std::size_t>(problem_size_)) {
            throw std::invalid_argument("demand has incorrect shape");
        }
        if (windows_.size() != static_cast<std::size_t>(problem_size_ * 2)) {
            throw std::invalid_argument("windows must have shape (problem_size, 2)");
        }

        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
            if (min_ <= 1e-9) {
                throw std::invalid_argument("min must be greater than 1e-9 when min_max is enabled");
            }
            max_ = -1.0;
        }

        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(problem_size_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(problem_size_ * problem_size_), 1.0);
            if (min_max_) {
                std::fill(pheromone_.begin(), pheromone_.end(), min_);
            }
        }

        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(problem_size_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(problem_size_ * problem_size_));
            for (std::int64_t row = 0; row < problem_size_; ++row) {
                for (std::int64_t col = 0; col < problem_size_; ++col) {
                    heuristic_[index(row, col)] = 1.0 / std::max(distances_[index(row, col)], kEps);
                }
            }
        }
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }

    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(pheromone_.size()))) != problem_size_) {
            throw std::invalid_argument("pheromone has incorrect shape");
        }
    }

    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        heuristic_ = read_square_matrix(heuristic_array);
        if (static_cast<std::int64_t>(std::sqrt(static_cast<double>(heuristic_.size()))) != problem_size_) {
            throw std::invalid_argument("heuristic has incorrect shape");
        }
    }

    py::array_t<double> get_pheromone() const { return matrix_to_numpy(pheromone_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy(heuristic_); }
    py::array_t<double> get_distances() const { return matrix_to_numpy(distances_); }
    py::array_t<double> get_demand() const { return vector_to_numpy(demand_); }
    py::array_t<double> get_windows() const { return windows_to_numpy(); }
    std::vector<std::int64_t> get_shortest_path() const { return shortest_path_; }
    double get_lowest_cost() const { return lowest_cost_; }

    py::tuple sample_trace() {
        auto sampled = sample_paths();
        py::array_t<double> costs = vector_to_numpy(sampled.costs);
        return py::make_tuple(costs, sampled.trace);
    }

    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        std::int64_t seq_len = 0;
        auto paths = read_paths(paths_array, seq_len);
        return vector_to_numpy(gen_path_costs_vector(paths, seq_len));
    }

    void update_pheromone_from_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        std::int64_t seq_len = 0;
        auto paths = read_paths(paths_array, seq_len);
        auto costs = read_costs(costs_array);
        update_pheromone(paths, seq_len, costs);
    }

    double run(int n_iterations) {
        if (n_iterations <= 0) {
            return lowest_cost_;
        }
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
        }
        return lowest_cost_;
    }

private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        RouteTrace trace;
    };

    std::vector<double> distances_;
    std::vector<double> demand_;
    std::vector<double> windows_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    double service_time_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    double min_ = 0.0;
    double max_ = -1.0;
    double capacity_;
    std::int64_t problem_size_;
    std::vector<std::int64_t> shortest_path_;
    double lowest_cost_ = std::numeric_limits<double>::infinity();
    std::mt19937_64 rng_;

    std::size_t index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * problem_size_ + col);
    }

    std::size_t window_index(std::int64_t node, int side) const {
        return static_cast<std::size_t>(node * 2 + side);
    }

    py::array_t<double> matrix_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out({problem_size_, problem_size_});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < problem_size_; ++row) {
            for (std::int64_t col = 0; col < problem_size_; ++col) {
                buf(row, col) = values[index(row, col)];
            }
        }
        return out;
    }

    py::array_t<double> vector_to_numpy(const std::vector<double>& values) const {
        py::array_t<double> out(values.size());
        auto buf = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = values[i];
        }
        return out;
    }

    py::array_t<double> windows_to_numpy() const {
        py::array_t<double> out(std::vector<py::ssize_t>{problem_size_, 2});
        auto buf = out.mutable_unchecked<2>();
        for (std::int64_t row = 0; row < problem_size_; ++row) {
            buf(row, 0) = windows_[window_index(row, 0)];
            buf(row, 1) = windows_[window_index(row, 1)];
        }
        return out;
    }

    std::vector<std::int64_t> read_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        std::int64_t& seq_len) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (seq_len, n_ants)");
        }
        seq_len = buf.shape[0];
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + seq_len * n_ants_);
    }

    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) const {
        auto buf = costs_array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }

    void update_visit_mask(std::vector<unsigned char>& visit_mask, const std::vector<std::int64_t>& actions) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(actions[static_cast<std::size_t>(ant)])] = 0;
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_)] = 1;
            if (actions[static_cast<std::size_t>(ant)] == 0) {
                bool has_remaining = false;
                for (std::int64_t node = 1; node < problem_size_; ++node) {
                    if (visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] != 0) {
                        has_remaining = true;
                        break;
                    }
                }
                if (has_remaining) {
                    visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_)] = 0;
                }
            }
        }
    }

    void update_capacity_mask(
        const std::vector<std::int64_t>& actions,
        std::vector<double>& used_capacity,
        std::vector<unsigned char>& capacity_mask) const {
        std::fill(capacity_mask.begin(), capacity_mask.end(), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            if (actions[static_cast<std::size_t>(ant)] == 0) {
                used_capacity[static_cast<std::size_t>(ant)] = 0.0;
            }
            used_capacity[static_cast<std::size_t>(ant)] += demand_[static_cast<std::size_t>(actions[static_cast<std::size_t>(ant)])];
            const double remaining = capacity_ - used_capacity[static_cast<std::size_t>(ant)];
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                if (demand_[static_cast<std::size_t>(node)] > remaining + 1e-10) {
                    capacity_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] = 0;
                }
            }
        }
    }

    void update_time_window_mask(
        const std::vector<std::int64_t>& actions,
        std::vector<double>& current_time,
        const std::vector<std::int64_t>* prev_actions,
        std::vector<unsigned char>& time_window_mask) const {
        std::fill(time_window_mask.begin(), time_window_mask.end(), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto action = actions[static_cast<std::size_t>(ant)];
            if (prev_actions != nullptr) {
                const auto prev = (*prev_actions)[static_cast<std::size_t>(ant)];
                current_time[static_cast<std::size_t>(ant)] += distances_[index(prev, action)];
                current_time[static_cast<std::size_t>(ant)] =
                    std::max(current_time[static_cast<std::size_t>(ant)], windows_[window_index(action, 0)]);
                current_time[static_cast<std::size_t>(ant)] += service_time_;
            }
            if (action == 0) {
                current_time[static_cast<std::size_t>(ant)] = 0.0;
            }

            for (std::int64_t node = 0; node < problem_size_; ++node) {
                const double arrive = current_time[static_cast<std::size_t>(ant)] + distances_[index(action, node)];
                if (arrive > windows_[window_index(node, 1)] + 1e-10) {
                    time_window_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] = 0;
                    continue;
                }
                const double start_service = std::max(arrive, windows_[window_index(node, 0)]);
                const double depot_arrival = start_service + service_time_ + distances_[index(node, 0)];
                if (depot_arrival > windows_[window_index(0, 1)] + 1e-10) {
                    time_window_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] = 0;
                }
            }
            time_window_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_)] = 1;
        }
    }

    bool check_done(const std::vector<unsigned char>& visit_mask, const std::vector<std::int64_t>& actions) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            if (actions[static_cast<std::size_t>(ant)] != 0) {
                return false;
            }
            for (std::int64_t node = 1; node < problem_size_; ++node) {
                if (visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node)] != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<std::int64_t> pick_move(
        const std::vector<std::int64_t>& prev,
        const std::vector<unsigned char>& visit_mask,
        const std::vector<unsigned char>& capacity_mask,
        const std::vector<unsigned char>& time_window_mask,
        std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        std::vector<double> weights(static_cast<std::size_t>(problem_size_));

        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto prev_node = prev[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node);
                if (visit_mask[idx] == 0 || capacity_mask[idx] == 0 || time_window_mask[idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double pheromone_term = std::pow(std::max(pheromone_[index(prev_node, node)], kEps), alpha_);
                const double heuristic_term = std::pow(std::max(heuristic_[index(prev_node, node)], kEps), beta_);
                const double weight = pheromone_term * heuristic_term;
                weights[static_cast<std::size_t>(node)] = weight;
                normalizer += weight;
            }

            const double threshold = unit_dist(rng) * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                if (weights[static_cast<std::size_t>(node)] > 0.0) {
                    last_feasible = node;
                }
                cumulative += weights[static_cast<std::size_t>(node)];
                if (cumulative > threshold) {
                    chosen = node;
                    break;
                }
            }
            if (chosen < 0) {
                chosen = last_feasible >= 0 ? last_feasible : 0;
            }
            actions[static_cast<std::size_t>(ant)] = chosen;
        }
        return actions;
    }

    SampledRoutes sample_paths() {
        std::vector<std::vector<std::int64_t>> path_steps;
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_visit_mask(visit_mask, actions);

        std::vector<double> used_capacity(static_cast<std::size_t>(n_ants_), 0.0);
        std::vector<unsigned char> capacity_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_capacity_mask(actions, used_capacity, capacity_mask);

        std::vector<double> current_time(static_cast<std::size_t>(n_ants_), 0.0);
        std::vector<unsigned char> time_window_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_time_window_mask(actions, current_time, nullptr, time_window_mask);

        path_steps.push_back(actions);
        while (!check_done(visit_mask, actions)) {
            auto prev_actions = actions;
            actions = pick_move(actions, visit_mask, capacity_mask, time_window_mask, rng_);
            path_steps.push_back(actions);
            update_visit_mask(visit_mask, actions);
            update_capacity_mask(actions, used_capacity, capacity_mask);
            update_time_window_mask(actions, current_time, &prev_actions, time_window_mask);
        }

        const auto seq_len = static_cast<std::int64_t>(path_steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = path_steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }

        auto costs = gen_path_costs_vector(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(costs), std::move(trace)};
    }

    std::vector<double> gen_path_costs_vector(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double cost = 0.0;
            for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
                const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
                cost += distances_[index(curr, next)];
            }
            costs[static_cast<std::size_t>(ant)] = cost;
        }
        return costs;
    }

    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        auto best_iter = std::min_element(costs.begin(), costs.end());
        if (best_iter == costs.end()) {
            return;
        }
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const double best_cost = *best_iter;
        if (best_cost >= lowest_cost_) {
            return;
        }

        lowest_cost_ = best_cost;
        shortest_path_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            shortest_path_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }

        if (min_max_) {
            const double next_max = static_cast<double>(problem_size_) / std::max(lowest_cost_, kEps);
            if (max_ < 0.0) {
                const double pheromone_max = *std::max_element(pheromone_.begin(), pheromone_.end());
                if (pheromone_max > 0.0) {
                    const double scale = next_max / pheromone_max;
                    for (auto& value : pheromone_) {
                        value *= scale;
                    }
                }
            }
            max_ = next_max;
        }
    }

    void update_pheromone(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }

        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            const double deposit = 1.0 / std::max(*best_iter, kEps);
            deposit_path(paths_flat, seq_len, best_idx, deposit);
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                const double deposit = 1.0 / std::max(costs[static_cast<std::size_t>(ant)], kEps);
                deposit_path(paths_flat, seq_len, ant, deposit);
            }
        }

        if (min_max_) {
            for (auto& value : pheromone_) {
                if (value > 1e-9 && value < min_) {
                    value = min_;
                }
                if (max_ > 0.0 && value > max_) {
                    value = max_;
                }
            }
        }

        for (auto& value : pheromone_) {
            if (value < 1e-10) {
                value = 1e-10;
            }
        }
    }

    void deposit_path(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

#include "aco_more.inc"

PYBIND11_MODULE(alphaant_tsp_aco_cpp, m) {
    py::class_<BatchTrace>(m, "BatchTrace")
        .def_readonly("paths_flat", &BatchTrace::paths_flat)
        .def_readonly("problem_size", &BatchTrace::problem_size)
        .def_readonly("n_ants", &BatchTrace::n_ants)
        .def("paths_numpy", &BatchTrace::paths_numpy);

    py::class_<MultiBatchTrace>(m, "MultiBatchTrace")
        .def_readonly("n_colonies", &MultiBatchTrace::n_colonies)
        .def_readonly("problem_size", &MultiBatchTrace::problem_size)
        .def_readonly("n_ants", &MultiBatchTrace::n_ants)
        .def("paths_numpy", &MultiBatchTrace::paths_numpy);

    py::class_<RouteTrace>(m, "RouteTrace")
        .def_readonly("seq_len", &RouteTrace::seq_len)
        .def_readonly("n_ants", &RouteTrace::n_ants)
        .def("paths_numpy", &RouteTrace::paths_numpy);

    py::class_<ACO_TSP>(m, "ACO_TSP")
        .def(
            py::init<
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                int,
                double,
                double,
                double,
                bool,
                bool,
                py::object,
                py::object,
                py::object,
                std::uint64_t>(),
            py::arg("distances"),
            py::arg("n_ants") = 20,
            py::arg("decay") = 0.9,
            py::arg("alpha") = 1.0,
            py::arg("beta") = 1.0,
            py::arg("elitist") = false,
            py::arg("min_max") = false,
            py::arg("pheromone") = py::none(),
            py::arg("heuristic") = py::none(),
            py::arg("min") = py::none(),
            py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_TSP::set_seed)
        .def("set_pheromone", &ACO_TSP::set_pheromone)
        .def("set_heuristic", &ACO_TSP::set_heuristic)
        .def("get_pheromone", &ACO_TSP::get_pheromone)
        .def("get_heuristic", &ACO_TSP::get_heuristic)
        .def("get_distances", &ACO_TSP::get_distances)
        .def("get_shortest_path", &ACO_TSP::get_shortest_path)
        .def("get_lowest_cost", &ACO_TSP::get_lowest_cost)
        .def("sparsify", &ACO_TSP::sparsify)
        .def("sample_trace", &ACO_TSP::sample_trace)
        .def("sample_ant_heuristic_trace", &ACO_TSP::sample_ant_heuristic_trace, py::arg("ant_heuristics"), py::arg("seed") = std::random_device{}())
        .def("sample_many_traces", &ACO_TSP::sample_many_traces, py::arg("heuristics"), py::arg("seed") = std::random_device{}())
        .def("gen_path_costs", &ACO_TSP::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_TSP::update_pheromone_from_paths)
        .def("run_many", &ACO_TSP::run_many, py::arg("heuristics"), py::arg("n_iterations"), py::arg("seed") = std::random_device{}())
        .def("run_with_ant_heuristics", &ACO_TSP::run_with_ant_heuristics, py::arg("ant_heuristics"), py::arg("n_iterations"), py::arg("seed") = std::random_device{}())
        .def("run", &ACO_TSP::run);

    py::class_<ACO_CVRP>(m, "ACO_CVRP")
        .def(
            py::init<
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                int,
                double,
                double,
                double,
                bool,
                bool,
                py::object,
                py::object,
                py::object,
                double,
                std::uint64_t>(),
            py::arg("distances"),
            py::arg("demand"),
            py::arg("n_ants") = 20,
            py::arg("decay") = 0.9,
            py::arg("alpha") = 1.0,
            py::arg("beta") = 1.0,
            py::arg("elitist") = false,
            py::arg("min_max") = false,
            py::arg("pheromone") = py::none(),
            py::arg("heuristic") = py::none(),
            py::arg("min") = py::none(),
            py::arg("capacity") = 50.0,
            py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_CVRP::set_seed)
        .def("set_pheromone", &ACO_CVRP::set_pheromone)
        .def("set_heuristic", &ACO_CVRP::set_heuristic)
        .def("get_pheromone", &ACO_CVRP::get_pheromone)
        .def("get_heuristic", &ACO_CVRP::get_heuristic)
        .def("get_distances", &ACO_CVRP::get_distances)
        .def("get_demand", &ACO_CVRP::get_demand)
        .def("get_shortest_path", &ACO_CVRP::get_shortest_path)
        .def("get_lowest_cost", &ACO_CVRP::get_lowest_cost)
        .def("sample_trace", &ACO_CVRP::sample_trace)
        .def("sample_ant_heuristic_trace", &ACO_CVRP::sample_ant_heuristic_trace, py::arg("ant_heuristics"), py::arg("seed") = std::random_device{}())
        .def("gen_path_costs", &ACO_CVRP::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_CVRP::update_pheromone_from_paths)
        .def("run_with_ant_heuristics", &ACO_CVRP::run_with_ant_heuristics, py::arg("ant_heuristics"), py::arg("n_iterations"), py::arg("seed") = std::random_device{}())
        .def("run", &ACO_CVRP::run);

    py::class_<ACO_VRPTW>(m, "ACO_VRPTW")
        .def(
            py::init<
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                const py::array_t<double, py::array::c_style | py::array::forcecast>&,
                double,
                int,
                double,
                double,
                double,
                bool,
                bool,
                py::object,
                py::object,
                py::object,
                double,
                std::uint64_t>(),
            py::arg("distances"),
            py::arg("demand"),
            py::arg("windows"),
            py::arg("service_time") = 0.0,
            py::arg("n_ants") = 20,
            py::arg("decay") = 0.9,
            py::arg("alpha") = 1.0,
            py::arg("beta") = 1.0,
            py::arg("elitist") = false,
            py::arg("min_max") = false,
            py::arg("pheromone") = py::none(),
            py::arg("heuristic") = py::none(),
            py::arg("min") = py::none(),
            py::arg("capacity") = 1.0,
            py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_VRPTW::set_seed)
        .def("set_pheromone", &ACO_VRPTW::set_pheromone)
        .def("set_heuristic", &ACO_VRPTW::set_heuristic)
        .def("get_pheromone", &ACO_VRPTW::get_pheromone)
        .def("get_heuristic", &ACO_VRPTW::get_heuristic)
        .def("get_distances", &ACO_VRPTW::get_distances)
        .def("get_demand", &ACO_VRPTW::get_demand)
        .def("get_windows", &ACO_VRPTW::get_windows)
        .def("get_shortest_path", &ACO_VRPTW::get_shortest_path)
        .def("get_lowest_cost", &ACO_VRPTW::get_lowest_cost)
        .def("sample_trace", &ACO_VRPTW::sample_trace)
        .def("gen_path_costs", &ACO_VRPTW::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_VRPTW::update_pheromone_from_paths)
        .def("run", &ACO_VRPTW::run);

    bind_additional_aco_classes(m);
}
