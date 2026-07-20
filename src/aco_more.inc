namespace {

std::vector<double> read_matrix_2d(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
    std::int64_t& rows,
    std::int64_t& cols) {
    auto buf = array.request();
    if (buf.ndim != 2) {
        throw std::invalid_argument("expected a rank-2 matrix");
    }
    rows = static_cast<std::int64_t>(buf.shape[0]);
    cols = static_cast<std::int64_t>(buf.shape[1]);
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + rows * cols);
}

py::array_t<double> vector_to_numpy_generic(const std::vector<double>& values) {
    py::array_t<double> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = values[i];
    }
    return out;
}

py::array_t<double> matrix_to_numpy_generic(const std::vector<double>& values, std::int64_t rows, std::int64_t cols) {
    py::array_t<double> out({rows, cols});
    auto buf = out.mutable_unchecked<2>();
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            buf(row, col) = values[static_cast<std::size_t>(row * cols + col)];
        }
    }
    return out;
}

py::array_t<std::int64_t> vector_to_numpy_int64(const std::vector<std::int64_t>& values) {
    py::array_t<std::int64_t> out(values.size());
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < values.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = values[i];
    }
    return out;
}

py::array_t<std::int64_t> matrix_to_numpy_int64(const std::vector<std::int64_t>& values, std::int64_t rows, std::int64_t cols) {
    py::array_t<std::int64_t> out({rows, cols});
    auto buf = out.mutable_unchecked<2>();
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t col = 0; col < cols; ++col) {
            buf(row, col) = values[static_cast<std::size_t>(row * cols + col)];
        }
    }
    return out;
}

class ACO_BPP {
public:
    ACO_BPP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& demand_array,
        int n_ants = 20,
        double decay = 0.9,
        double alpha = 1.0,
        double beta = 1.0,
        bool elitist = false,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        double capacity = 150.0,
        std::uint64_t seed = std::random_device{}())
        : demand_(read_vector(demand_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          capacity_(capacity),
          problem_size_(static_cast<std::int64_t>(demand_array.shape(0))),
          rng_(seed) {
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(problem_size_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(problem_size_ * problem_size_), 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(problem_size_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(problem_size_ * problem_size_));
            for (std::int64_t row = 0; row < problem_size_; ++row) {
                for (std::int64_t col = 0; col < problem_size_; ++col) {
                    heuristic_[index(row, col)] = demand_[static_cast<std::size_t>(col)];
                }
                heuristic_[index(row, 0)] = 1e-5;
            }
        }
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
    }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        heuristic_ = read_square_matrix(heuristic_array);
    }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, problem_size_, problem_size_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, problem_size_, problem_size_); }
    py::array_t<double> get_demand() const { return vector_to_numpy_generic(demand_); }
    std::vector<std::int64_t> get_shortest_path() const { return shortest_path_; }
    double get_best_fitness() const { return best_fitness_; }
    double get_lowest_cost() const { return best_fitness_; }

    py::tuple sample_trace() {
        auto sampled = sample_paths();
        return py::make_tuple(vector_to_numpy_generic(sampled.costs), sampled.trace);
    }

    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        std::int64_t seq_len = 0;
        auto paths = read_paths(paths_array, seq_len);
        return vector_to_numpy_generic(gen_path_costs_vector(paths, seq_len));
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
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
        }
        return best_fitness_;
    }

private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        RouteTrace trace;
    };

    std::vector<double> demand_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    double capacity_;
    std::int64_t problem_size_;
    std::vector<std::int64_t> shortest_path_;
    double best_fitness_ = 0.0;
    std::mt19937_64 rng_;

    std::size_t index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * problem_size_ + col);
    }

    std::vector<std::int64_t> read_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        std::int64_t& seq_len) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (seq_len, n_ants)");
        }
        seq_len = static_cast<std::int64_t>(buf.shape[0]);
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
        const std::vector<unsigned char>& capacity_mask) {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> actions(static_cast<std::size_t>(n_ants_), 0);
        // Pre-draw the per-ant uniform variates SERIALLY so the RNG stream (and
        // therefore results) are bit-identical to the serial version regardless of
        // thread count; the heavy per-ant weight computation below is then run in
        // parallel over ants (each ant is independent) with a thread-local buffer.
        std::vector<double> rand_vals(static_cast<std::size_t>(n_ants_));
        for (int ant = 0; ant < n_ants_; ++ant) {
            rand_vals[static_cast<std::size_t>(ant)] = unit_dist(rng_);
        }
        #pragma omp parallel for schedule(static) if (n_ants_ > 1)
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<double> weights(static_cast<std::size_t>(problem_size_), 0.0);
            const auto prev_node = prev[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t node = 0; node < problem_size_; ++node) {
                const auto mask_idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(problem_size_) + static_cast<std::size_t>(node);
                if (visit_mask[mask_idx] == 0 || capacity_mask[mask_idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double value = std::pow(std::max(pheromone_[index(prev_node, node)], kEps), alpha_) *
                                     std::pow(std::max(heuristic_[index(prev_node, node)], kEps), beta_);
                weights[static_cast<std::size_t>(node)] = value;
                normalizer += value;
            }
            const double threshold = rand_vals[static_cast<std::size_t>(ant)] * normalizer;
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
        std::vector<double> used_capacity(static_cast<std::size_t>(n_ants_), 0.0);
        std::vector<unsigned char> capacity_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(problem_size_), 1);
        update_visit_mask(visit_mask, actions);
        update_capacity_mask(actions, used_capacity, capacity_mask);
        path_steps.push_back(actions);
        while (!check_done(visit_mask, actions)) {
            actions = pick_move(actions, visit_mask, capacity_mask);
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
            std::int64_t trailing_zeros = 0;
            for (std::int64_t step = seq_len - 1; step >= 0; --step) {
                if (paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] == 0) {
                    ++trailing_zeros;
                } else {
                    break;
                }
                if (step == 0) {
                    break;
                }
            }
            const double n_bins = static_cast<double>(seq_len - trailing_zeros - problem_size_ + 1);
            double fitness = 0.0;
            double sub_fitness = 0.0;
            for (std::int64_t step = 1; step < seq_len; ++step) {
                const auto node = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                if (node != 0) {
                    sub_fitness += demand_[static_cast<std::size_t>(node)];
                } else {
                    fitness += std::pow(sub_fitness / capacity_, 2.0);
                    sub_fitness = 0.0;
                }
            }
            costs[static_cast<std::size_t>(ant)] = -fitness / std::max(n_bins, 1.0);
        }
        return costs;
    }

    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        auto best_iter = std::min_element(costs.begin(), costs.end());
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const double best_fit = -(*best_iter);
        if (best_fit <= best_fitness_) {
            return;
        }
        best_fitness_ = best_fit;
        shortest_path_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            shortest_path_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
    }

    void update_pheromone(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (elitist_) {
            auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            const double best_fit = -(*best_iter);
            deposit_path(paths_flat, seq_len, best_idx, best_fit);
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, seq_len, ant, -costs[static_cast<std::size_t>(ant)] / static_cast<double>(n_ants_));
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

class ACO_MKP {
public:
    ACO_MKP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& prize_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& weight_array,
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
        : prize_(read_vector(prize_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          n_(static_cast<std::int64_t>(prize_array.shape(0))),
          rng_(seed) {
        std::int64_t rows = 0;
        std::int64_t cols = 0;
        weight_ = read_matrix_2d(weight_array, rows, cols);
        if (rows != n_) {
            throw std::invalid_argument("weight has incorrect shape");
        }
        m_ = cols;
        q_ = 1.0 / std::accumulate(prize_.begin(), prize_.end(), 0.0);
        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
            max_ = 20.0;
        }

        const auto expanded_dim = static_cast<std::size_t>(n_ + 1);
        pheromone_ = read_optional_square_matrix(pheromone_obj, expanded_dim);
        if (pheromone_.empty()) {
            pheromone_.assign(expanded_dim * expanded_dim, min_max_ ? min_ : 1.0);
        }

        auto heuristic_raw = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_));
        if (heuristic_raw.empty()) {
            heuristic_raw.resize(static_cast<std::size_t>(n_ * n_));
            std::vector<double> ratio(static_cast<std::size_t>(n_), 0.0);
            for (std::int64_t item = 0; item < n_; ++item) {
                double sum_weight = 0.0;
                for (std::int64_t dim = 0; dim < m_; ++dim) {
                    sum_weight += weight_value(item, dim);
                }
                ratio[static_cast<std::size_t>(item)] = prize_[static_cast<std::size_t>(item)] / std::max(sum_weight, kEps);
            }
            for (std::int64_t row = 0; row < n_; ++row) {
                for (std::int64_t col = 0; col < n_; ++col) {
                    heuristic_raw[static_cast<std::size_t>(row * n_ + col)] = ratio[static_cast<std::size_t>(col)];
                }
            }
        }

        prize_.push_back(0.0);
        const auto old_weight_size = weight_.size();
        weight_.resize(old_weight_size + static_cast<std::size_t>(m_), 0.0);

        heuristic_.assign(expanded_dim * expanded_dim, 1e-10);
        for (std::int64_t row = 0; row < n_; ++row) {
            for (std::int64_t col = 0; col < n_; ++col) {
                heuristic_[expanded_index(row, col)] = heuristic_raw[static_cast<std::size_t>(row * n_ + col)];
            }
        }
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
    }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        auto raw = read_square_matrix(heuristic_array);
        if (static_cast<std::size_t>(raw.size()) == static_cast<std::size_t>(n_ * n_)) {
            const auto expanded_dim = static_cast<std::size_t>(n_ + 1);
            heuristic_.assign(expanded_dim * expanded_dim, 1e-10);
            for (std::int64_t row = 0; row < n_; ++row) {
                for (std::int64_t col = 0; col < n_; ++col) {
                    heuristic_[expanded_index(row, col)] = raw[static_cast<std::size_t>(row * n_ + col)];
                }
            }
        } else {
            heuristic_ = std::move(raw);
        }
    }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_ + 1, n_ + 1); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_ + 1, n_ + 1); }

    py::tuple sample_trace() {
        auto sampled = sample_solutions();
        return py::make_tuple(vector_to_numpy_generic(sampled.objs), sampled.trace);
    }

    py::array_t<double> gen_sol_obj(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array) const {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        return vector_to_numpy_generic(gen_obj_vector(sols, seq_len));
    }

    void update_pheromone_from_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& objs_array) {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        auto objs = read_costs(objs_array);
        update_pheromone(sols, seq_len, objs);
    }
    double get_lowest_cost() const { return best_obj_; }

    py::tuple run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_solutions();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.objs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.objs);
        }
        return py::make_tuple(best_obj_, best_sol_);
    }

private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> objs;
        RouteTrace trace;
    };

    std::vector<double> prize_;
    std::vector<double> weight_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    std::int64_t n_;
    std::int64_t m_;
    double min_ = 0.0;
    double max_ = 20.0;
    double q_ = 0.0;
    double best_obj_ = 0.0;
    std::vector<std::int64_t> best_sol_;
    std::mt19937_64 rng_;

    std::size_t weight_index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * m_ + col);
    }
    double weight_value(std::int64_t row, std::int64_t col) const {
        return weight_[weight_index(row, col)];
    }
    std::size_t expanded_index(std::int64_t row, std::int64_t col) const {
        return static_cast<std::size_t>(row * (n_ + 1) + col);
    }
    std::vector<std::int64_t> read_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array,
        std::int64_t& seq_len) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("solutions must have shape (seq_len, n_ants)");
        }
        seq_len = static_cast<std::int64_t>(buf.shape[0]);
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + seq_len * n_ants_);
    }
    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& values_array) const {
        auto buf = values_array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("values must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }

    void update_knapsack(
        std::vector<unsigned char>& mask,
        std::vector<double>& knapsack,
        const std::vector<std::int64_t>& new_item) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto item = new_item[static_cast<std::size_t>(ant)];
            mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(item)] = 0;
            for (std::int64_t dim = 0; dim < m_; ++dim) {
                knapsack[static_cast<std::size_t>(ant) * static_cast<std::size_t>(m_) + static_cast<std::size_t>(dim)] += weight_value(item, dim);
            }
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<std::int64_t> candidates;
            for (std::int64_t item = 0; item <= n_; ++item) {
                if (mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(item)] != 0) {
                    candidates.push_back(item);
                }
            }
            if (candidates.size() > 1) {
                for (const auto candidate : candidates) {
                    bool infeasible = false;
                    for (std::int64_t dim = 0; dim < m_; ++dim) {
                        const double next_weight = knapsack[static_cast<std::size_t>(ant) * static_cast<std::size_t>(m_) + static_cast<std::size_t>(dim)] + weight_value(candidate, dim);
                        if (next_weight > static_cast<double>(n_ / 2)) {
                            infeasible = true;
                            break;
                        }
                    }
                    if (infeasible) {
                        mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(candidate)] = 0;
                    }
                }
            }
            mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(n_)] = 1;
        }
    }

    void update_dummy_state(const std::vector<unsigned char>& mask, std::vector<unsigned char>& dummy_mask) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            bool finished = true;
            for (std::int64_t item = 0; item < n_; ++item) {
                if (mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(item)] != 0) {
                    finished = false;
                    break;
                }
            }
            if (finished) {
                dummy_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(n_)] = 1;
            }
        }
    }

    bool check_done(const std::vector<unsigned char>& mask) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            for (std::int64_t item = 0; item < n_; ++item) {
                if (mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(item)] != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<std::int64_t> pick_item(
        const std::vector<std::int64_t>& items,
        const std::vector<unsigned char>& mask,
        const std::vector<unsigned char>& dummy_mask) {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> next(static_cast<std::size_t>(n_ants_), 0);
        // Pre-draw per-ant variates serially (bit-identical to serial), then run the
        // heavy per-ant weight computation in parallel with a thread-local buffer.
        std::vector<double> rand_vals(static_cast<std::size_t>(n_ants_));
        for (int ant = 0; ant < n_ants_; ++ant) {
            rand_vals[static_cast<std::size_t>(ant)] = unit_dist(rng_);
        }
        #pragma omp parallel for schedule(static) if (n_ants_ > 1)
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<double> weights(static_cast<std::size_t>(n_ + 1), 0.0);
            const auto prev = items[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t item = 0; item <= n_; ++item) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(item);
                if (mask[idx] == 0 || dummy_mask[idx] == 0) {
                    weights[static_cast<std::size_t>(item)] = 0.0;
                    continue;
                }
                const double value = std::pow(std::max(pheromone_[expanded_index(prev, item)], kEps), alpha_) *
                                     std::pow(std::max(heuristic_[expanded_index(prev, item)], kEps), beta_);
                weights[static_cast<std::size_t>(item)] = value;
                normalizer += value;
            }
            const double threshold = rand_vals[static_cast<std::size_t>(ant)] * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t item = 0; item <= n_; ++item) {
                if (weights[static_cast<std::size_t>(item)] > 0.0) {
                    last_feasible = item;
                }
                cumulative += weights[static_cast<std::size_t>(item)];
                if (cumulative > threshold) {
                    chosen = item;
                    break;
                }
            }
            if (chosen < 0) {
                chosen = last_feasible >= 0 ? last_feasible : 0;
            }
            next[static_cast<std::size_t>(ant)] = chosen;
        }
        return next;
    }

    SampledRoutes sample_solutions() {
        std::uniform_int_distribution<std::int64_t> start_dist(0, n_ - 1);
        std::vector<std::vector<std::int64_t>> steps;
        std::vector<std::int64_t> items(static_cast<std::size_t>(n_ants_), 0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            items[static_cast<std::size_t>(ant)] = start_dist(rng_);
        }
        steps.push_back(items);
        std::vector<double> knapsack(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(m_), 0.0);
        std::vector<unsigned char> mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_ + 1), 1);
        std::vector<unsigned char> dummy_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_ + 1), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            dummy_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(n_)] = 0;
        }
        update_knapsack(mask, knapsack, items);
        update_dummy_state(mask, dummy_mask);
        while (!check_done(mask)) {
            items = pick_item(items, mask, dummy_mask);
            steps.push_back(items);
            update_knapsack(mask, knapsack, items);
            update_dummy_state(mask, dummy_mask);
        }
        const auto seq_len = static_cast<std::int64_t>(steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }
        auto objs = gen_obj_vector(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(objs), std::move(trace)};
    }

    std::vector<double> gen_obj_vector(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len) const {
        std::vector<double> objs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double total = 0.0;
            for (std::int64_t step = 0; step < seq_len; ++step) {
                total += prize_[static_cast<std::size_t>(paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)])];
            }
            objs[static_cast<std::size_t>(ant)] = total;
        }
        return objs;
    }

    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& objs) {
        auto best_iter = std::max_element(objs.begin(), objs.end());
        const auto best_idx = static_cast<int>(std::distance(objs.begin(), best_iter));
        if (*best_iter <= best_obj_) {
            return;
        }
        best_obj_ = *best_iter;
        best_sol_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            best_sol_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
    }

    void update_pheromone(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& objs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (elitist_) {
            const auto best_iter = std::max_element(objs.begin(), objs.end());
            const auto best_idx = static_cast<int>(std::distance(objs.begin(), best_iter));
            deposit_path(paths_flat, seq_len, best_idx, q_ * (*best_iter));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, seq_len, ant, q_ * objs[static_cast<std::size_t>(ant)]);
            }
        }
        if (min_max_) {
            for (auto& value : pheromone_) {
                if (value > 1e-9 && value < min_) {
                    value = min_;
                }
                if (value > max_) {
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
            pheromone_[expanded_index(curr, next)] += deposit;
        }
    }
};

class ACO_OP {
public:
    ACO_OP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& prizes_array,
        double max_len,
        int n_ants = 20,
        double decay = 0.9,
        double alpha = 1.0,
        double beta = 1.0,
        bool elitist = false,
        bool min_max = false,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        py::object min_obj = py::none(),
        py::object k_sparse_obj = py::none(),
        std::uint64_t seed = std::random_device{}())
        : distances_(read_square_matrix(distances_array)),
          prizes_(read_vector(prizes_array)),
          max_len_(max_len),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          n_(static_cast<std::int64_t>(prizes_array.shape(0))),
          rng_(seed) {
        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
        }
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(n_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(n_ * n_), min_max_ ? min_ : 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_));
        if (heuristic_.empty()) {
            if (k_sparse_obj.is_none()) {
                throw std::invalid_argument("k_sparse is required when heuristic is None");
            }
            sparsify(py::cast<int>(k_sparse_obj));
        }
        add_dummy_node();
    }

    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) {
        pheromone_ = read_square_matrix(pheromone_array);
    }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) {
        auto raw = read_square_matrix(heuristic_array);
        const auto raw_dim = static_cast<std::int64_t>(std::sqrt(static_cast<double>(raw.size())));
        if (raw_dim == n_) {
            heuristic_ = expand_heuristic_with_dummy(raw);
            return;
        }
        heuristic_ = std::move(raw);
    }
    double get_lowest_cost() const { return best_obj_; }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_ + 1, n_ + 1); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_ + 1, n_ + 1); }
    py::tuple sample_trace() {
        auto sampled = sample_solutions();
        return py::make_tuple(vector_to_numpy_generic(sampled.objs), sampled.trace);
    }
    py::tuple run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_solutions();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.objs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.objs);
        }
        return py::make_tuple(best_obj_, best_sol_);
    }
    py::array_t<double> gen_sol_obj(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array) const {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        return vector_to_numpy_generic(gen_obj_vector(sols, seq_len));
    }
    void update_pheromone_from_paths(
        const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& objs_array) {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        auto objs = read_costs(objs_array);
        update_pheromone(sols, seq_len, objs);
    }

private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> objs;
        RouteTrace trace;
    };
    std::vector<double> distances_;
    std::vector<double> prizes_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    double max_len_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    std::int64_t n_;
    double min_ = 0.1;
    double max_ = -1.0;
    double q_ = 0.0;
    double best_obj_ = 0.0;
    std::vector<std::int64_t> best_sol_;
    std::mt19937_64 rng_;
    std::size_t index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * (n_ + 1) + col); }
    std::size_t raw_index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * n_ + col); }
    void sparsify(int k_sparse) {
        std::vector<double> sparse(static_cast<std::size_t>(n_ * n_), 1e10);
        for (std::int64_t row = 0; row < n_; ++row) {
            std::vector<std::pair<double, std::int64_t>> edges;
            edges.reserve(static_cast<std::size_t>(n_));
            for (std::int64_t col = 0; col < n_; ++col) {
                edges.emplace_back(distances_[raw_index(row, col)], col);
            }
            std::partial_sort(edges.begin(), edges.begin() + k_sparse, edges.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            for (int i = 0; i < k_sparse; ++i) {
                sparse[raw_index(row, edges[static_cast<std::size_t>(i)].second)] = edges[static_cast<std::size_t>(i)].first;
            }
        }
        heuristic_.resize(sparse.size());
        for (std::size_t i = 0; i < sparse.size(); ++i) {
            const auto col = static_cast<std::int64_t>(i % static_cast<std::size_t>(n_));
            heuristic_[i] = prizes_[static_cast<std::size_t>(col)] / std::max(sparse[i], kEps);
        }
    }
    std::vector<double> expand_heuristic_with_dummy(const std::vector<double>& raw_heuristic) const {
        std::vector<double> expanded_heuristic(static_cast<std::size_t>((n_ + 1) * (n_ + 1)), 1.0);
        for (std::int64_t row = 0; row < n_; ++row) {
            for (std::int64_t col = 0; col < n_; ++col) {
                expanded_heuristic[index(row, col)] = raw_heuristic[raw_index(row, col)];
            }
            expanded_heuristic[index(n_, row)] = 0.0;
        }
        return expanded_heuristic;
    }
    void add_dummy_node() {
        std::vector<double> expanded_distances(static_cast<std::size_t>((n_ + 1) * (n_ + 1)), 0.0);
        for (std::int64_t row = 0; row < n_; ++row) {
            for (std::int64_t col = 0; col < n_; ++col) {
                expanded_distances[index(row, col)] = distances_[raw_index(row, col)];
            }
            expanded_distances[index(row, n_)] = 1e-10;
            expanded_distances[index(n_, row)] = 1e10;
        }
        expanded_distances[index(n_, n_)] = 1e-10;
        distances_ = std::move(expanded_distances);
        for (auto& value : distances_) {
            if (value == 1e-10) {
                value = 0.0;
            }
        }

        heuristic_ = expand_heuristic_with_dummy(heuristic_);
        pheromone_.assign(static_cast<std::size_t>((n_ + 1) * (n_ + 1)), 1.0);
        prizes_.push_back(0.0);
        q_ = 1.0 / std::accumulate(prizes_.begin(), prizes_.end(), 0.0);
    }
    std::vector<std::int64_t> read_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array, std::int64_t& seq_len) const {
        auto buf = paths_array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("solutions must have shape (seq_len, n_ants)");
        }
        seq_len = static_cast<std::int64_t>(buf.shape[0]);
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + seq_len * n_ants_);
    }
    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("values must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }
    bool check_done(const std::vector<unsigned char>& mask) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            for (std::int64_t node = 0; node < n_; ++node) {
                if (mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(node)] != 0) {
                    return false;
                }
            }
        }
        return true;
    }
    void update_mask(std::vector<double>& travel, const std::vector<std::int64_t>& cur, std::vector<unsigned char>& mask) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(cur[static_cast<std::size_t>(ant)])] = 0;
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto cur_node = cur[static_cast<std::size_t>(ant)];
            if (cur_node == n_) {
                continue;
            }
            for (std::int64_t node = 0; node <= n_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(node);
                if (mask[idx] == 0) {
                    continue;
                }
                const double trail = travel[static_cast<std::size_t>(ant)] + distances_[index(cur_node, node)] + distances_[index(node, 0)];
                if (trail > max_len_) {
                    mask[idx] = 0;
                }
            }
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(n_)] = 0;
            bool go_dummy = true;
            for (std::int64_t node = 0; node < n_; ++node) {
                if (mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(node)] != 0) {
                    go_dummy = false;
                    break;
                }
            }
            if (go_dummy) {
                mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(n_)] = 1;
            }
        }
    }
    std::vector<std::int64_t> pick_node(const std::vector<unsigned char>& mask, const std::vector<std::int64_t>& cur) {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> next(static_cast<std::size_t>(n_ants_), 0);
        // Pre-draw per-ant variates serially (bit-identical to serial), then run the
        // heavy per-ant weight computation in parallel with a thread-local buffer.
        std::vector<double> rand_vals(static_cast<std::size_t>(n_ants_));
        for (int ant = 0; ant < n_ants_; ++ant) {
            rand_vals[static_cast<std::size_t>(ant)] = unit_dist(rng_);
        }
        #pragma omp parallel for schedule(static) if (n_ants_ > 1)
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<double> weights(static_cast<std::size_t>(n_ + 1), 0.0);
            const auto prev = cur[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t node = 0; node <= n_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(node);
                if (mask[idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double value = std::pow(std::max(pheromone_[index(prev, node)], kEps), alpha_) *
                                     std::pow(std::max(heuristic_[index(prev, node)], kEps), beta_);
                weights[static_cast<std::size_t>(node)] = value;
                normalizer += value;
            }
            const double threshold = rand_vals[static_cast<std::size_t>(ant)] * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t node = 0; node <= n_; ++node) {
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
            next[static_cast<std::size_t>(ant)] = chosen;
        }
        return next;
    }
    SampledRoutes sample_solutions() {
        std::vector<std::vector<std::int64_t>> steps;
        std::vector<std::int64_t> cur(static_cast<std::size_t>(n_ants_), 0);
        std::vector<unsigned char> mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_ + 1), 1);
        std::vector<double> travel(static_cast<std::size_t>(n_ants_), 0.0);
        steps.push_back(cur);
        update_mask(travel, cur, mask);
        while (!check_done(mask)) {
            auto next = pick_node(mask, cur);
            steps.push_back(next);
            for (int ant = 0; ant < n_ants_; ++ant) {
                travel[static_cast<std::size_t>(ant)] += distances_[index(cur[static_cast<std::size_t>(ant)], next[static_cast<std::size_t>(ant)])];
            }
            cur = std::move(next);
            update_mask(travel, cur, mask);
        }
        const auto seq_len = static_cast<std::int64_t>(steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }
        auto objs = gen_obj_vector(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(objs), std::move(trace)};
    }
    std::vector<double> gen_obj_vector(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len) const {
        std::vector<double> objs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double total = 0.0;
            for (std::int64_t step = 0; step < seq_len; ++step) {
                total += prizes_[static_cast<std::size_t>(paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)])];
            }
            objs[static_cast<std::size_t>(ant)] = total;
        }
        return objs;
    }
    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& objs) {
        const auto best_iter = std::max_element(objs.begin(), objs.end());
        const auto best_idx = static_cast<int>(std::distance(objs.begin(), best_iter));
        if (*best_iter <= best_obj_) {
            return;
        }
        best_obj_ = *best_iter;
        best_sol_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            best_sol_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
        if (min_max_) {
            const double next_max = best_obj_ * static_cast<double>(n_) * q_;
            if (max_ < 0.0) {
                const double current_max = *std::max_element(pheromone_.begin(), pheromone_.end());
                for (auto& value : pheromone_) {
                    value *= next_max / std::max(current_max, kEps);
                }
            }
            max_ = next_max;
        }
    }
    void update_pheromone(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& objs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (elitist_) {
            const auto best_iter = std::max_element(objs.begin(), objs.end());
            const auto best_idx = static_cast<int>(std::distance(objs.begin(), best_iter));
            deposit_path(paths_flat, seq_len, best_idx, q_ * (*best_iter));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, seq_len, ant, q_ * objs[static_cast<std::size_t>(ant)]);
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
    }
    void deposit_path(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

class ACO_PCTSP {
public:
    ACO_PCTSP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& prizes_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& penalties_array,
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
          prizes_(read_vector(prizes_array)),
          penalties_(read_vector(penalties_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          n_(static_cast<std::int64_t>(prizes_array.shape(0))),
          min_prizes_(static_cast<double>(prizes_array.shape(0)) / 4.0),
          rng_(seed) {
        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
        }
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(n_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(n_ * n_), min_max_ ? min_ : 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(n_ * n_));
            for (std::int64_t row = 0; row < n_; ++row) {
                for (std::int64_t col = 0; col < n_; ++col) {
                    double dist = distances_[static_cast<std::size_t>(row * n_ + col)];
                    if (row == col) {
                        dist = 1e9;
                    }
                    heuristic_[static_cast<std::size_t>(row * n_ + col)] = (1e-10 + prizes_[static_cast<std::size_t>(col)]) / dist;
                }
            }
        }
    }
    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) { pheromone_ = read_square_matrix(pheromone_array); }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) { heuristic_ = read_square_matrix(heuristic_array); }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_, n_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_, n_); }
    double get_lowest_cost() const { return best_cost_; }
    py::tuple sample_trace() {
        auto sampled = sample_solutions();
        return py::make_tuple(vector_to_numpy_generic(sampled.costs), sampled.trace);
    }
    py::array_t<double> gen_sol_obj(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array) const {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        return vector_to_numpy_generic(gen_costs(sols, seq_len));
    }
    void update_pheromone_from_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& sols_array, const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        std::int64_t seq_len = 0;
        auto sols = read_paths(sols_array, seq_len);
        auto costs = read_costs(costs_array);
        update_pheromone(sols, seq_len, costs);
    }
    py::tuple run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_solutions();
            update_best(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.trace.seq_len, sampled.costs);
        }
        return py::make_tuple(best_cost_, best_sol_);
    }
private:
    struct SampledRoutes {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        RouteTrace trace;
    };
    std::vector<double> distances_;
    std::vector<double> prizes_;
    std::vector<double> penalties_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    std::int64_t n_;
    double min_prizes_;
    double min_ = 0.1;
    double max_ = -1.0;
    double best_cost_ = 1e10;
    std::vector<std::int64_t> best_sol_;
    std::mt19937_64 rng_;
    std::size_t index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * n_ + col); }
    std::vector<std::int64_t> read_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& array, std::int64_t& seq_len) const {
        auto buf = array.request();
        if (buf.ndim != 2 || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("solutions must have shape (seq_len, n_ants)");
        }
        seq_len = static_cast<std::int64_t>(buf.shape[0]);
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + seq_len * n_ants_);
    }
    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }
    bool check_done(const std::vector<std::int64_t>& cur) const {
        return std::all_of(cur.begin(), cur.end(), [](std::int64_t node) { return node == 0; });
    }
    void update_mask(
        std::vector<unsigned char>& visit_mask,
        std::vector<unsigned char>& depot_mask,
        const std::vector<std::int64_t>& cur,
        const std::vector<double>& collected) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(cur[static_cast<std::size_t>(ant)])] = 0;
        }
        for (int ant = 0; ant < n_ants_; ++ant) {
            const bool at_depot = cur[static_cast<std::size_t>(ant)] == 0;
            if (at_depot) {
                visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_)] = 1;
                for (std::int64_t node = 1; node < n_; ++node) {
                    visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node)] = 0;
                }
            } else {
                bool all_visited = true;
                for (std::int64_t node = 1; node < n_; ++node) {
                    if (visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node)] != 0) {
                        all_visited = false;
                        break;
                    }
                }
                if (collected[static_cast<std::size_t>(ant)] > min_prizes_ || all_visited) {
                    depot_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_)] = 1;
                }
            }
        }
    }
    std::vector<std::int64_t> pick_node(
        const std::vector<unsigned char>& visit_mask,
        const std::vector<unsigned char>& depot_mask,
        const std::vector<std::int64_t>& cur) {
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<std::int64_t> next(static_cast<std::size_t>(n_ants_), 0);
        std::vector<double> weights(static_cast<std::size_t>(n_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto prev = cur[static_cast<std::size_t>(ant)];
            double normalizer = 0.0;
            for (std::int64_t node = 0; node < n_; ++node) {
                const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node);
                if (visit_mask[idx] == 0 || depot_mask[idx] == 0) {
                    weights[static_cast<std::size_t>(node)] = 0.0;
                    continue;
                }
                const double value = std::pow(std::max(pheromone_[index(prev, node)], kEps), alpha_) *
                                     std::pow(std::max(heuristic_[index(prev, node)], kEps), beta_);
                weights[static_cast<std::size_t>(node)] = value;
                normalizer += value;
            }
            const double threshold = unit_dist(rng_) * normalizer;
            double cumulative = 0.0;
            std::int64_t chosen = -1;
            std::int64_t last_feasible = -1;
            for (std::int64_t node = 0; node < n_; ++node) {
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
            next[static_cast<std::size_t>(ant)] = chosen;
        }
        return next;
    }
    SampledRoutes sample_solutions() {
        std::vector<std::vector<std::int64_t>> steps;
        std::vector<std::int64_t> cur(static_cast<std::size_t>(n_ants_), 0);
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 1);
        std::vector<unsigned char> depot_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            depot_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_)] = 0;
        }
        std::vector<double> collected(static_cast<std::size_t>(n_ants_), 0.0);
        steps.push_back(cur);
        while (true) {
            auto next = pick_node(visit_mask, depot_mask, cur);
            steps.push_back(next);
            for (int ant = 0; ant < n_ants_; ++ant) {
                collected[static_cast<std::size_t>(ant)] += prizes_[static_cast<std::size_t>(next[static_cast<std::size_t>(ant)])];
            }
            cur = std::move(next);
            update_mask(visit_mask, depot_mask, cur, collected);
            if (check_done(cur)) {
                break;
            }
        }
        const auto seq_len = static_cast<std::int64_t>(steps.size());
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(n_ants_));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = steps[static_cast<std::size_t>(step)][static_cast<std::size_t>(ant)];
            }
        }
        auto costs = gen_costs(paths_flat, seq_len);
        RouteTrace trace;
        trace.paths_flat = paths_flat;
        trace.seq_len = seq_len;
        trace.n_ants = n_ants_;
        return SampledRoutes{std::move(paths_flat), std::move(costs), std::move(trace)};
    }
    std::vector<double> gen_costs(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<unsigned char> visited(static_cast<std::size_t>(n_), 0);
            double length = 0.0;
            for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
                const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
                length += distances_[index(curr, next)];
                visited[static_cast<std::size_t>(curr)] = 1;
            }
            visited[static_cast<std::size_t>(paths_flat[static_cast<std::size_t>((seq_len - 1) * n_ants_ + ant)])] = 1;
            double penalty = 0.0;
            for (std::int64_t node = 0; node < n_; ++node) {
                if (visited[static_cast<std::size_t>(node)] == 0) {
                    penalty += penalties_[static_cast<std::size_t>(node)];
                }
            }
            costs[static_cast<std::size_t>(ant)] = length + penalty;
        }
        return costs;
    }
    void update_best(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, const std::vector<double>& costs) {
        const auto best_iter = std::min_element(costs.begin(), costs.end());
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const double best_obj = *best_iter;
        if (best_obj >= best_cost_) {
            return;
        }
        best_cost_ = best_obj;
        best_sol_.resize(static_cast<std::size_t>(seq_len));
        for (std::int64_t step = 0; step < seq_len; ++step) {
            best_sol_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
        if (min_max_) {
            const double next_max = static_cast<double>(n_ - 1) / std::max(best_cost_, kEps);
            if (max_ < 0.0) {
                const double current_max = *std::max_element(pheromone_.begin(), pheromone_.end());
                for (auto& value : pheromone_) {
                    value *= next_max / std::max(current_max, kEps);
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
            deposit_path(paths_flat, seq_len, best_idx, 1.0 / std::max(*best_iter, kEps));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, seq_len, ant, 1.0 / std::max(costs[static_cast<std::size_t>(ant)], kEps));
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
    }
    void deposit_path(const std::vector<std::int64_t>& paths_flat, std::int64_t seq_len, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < seq_len; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

class ACO_SMTWTP {
public:
    ACO_SMTWTP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& due_time_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& weights_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& processing_time_array,
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
        : due_time_(read_vector(due_time_array)),
          weights_(read_vector(weights_array)),
          processing_time_(read_vector(processing_time_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          n_(static_cast<std::int64_t>(due_time_array.shape(0))),
          rng_(seed) {
        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
            max_ = 1.0;
        }
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(n_ + 1));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>((n_ + 1) * (n_ + 1)), min_max_ ? min_ : 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_ + 1));
        if (heuristic_.empty()) {
            heuristic_.assign(static_cast<std::size_t>((n_ + 1) * (n_ + 1)), 1.0);
            for (std::int64_t row = 0; row <= n_; ++row) {
                heuristic_[index(row, 0)] = 1.0;
                for (std::int64_t col = 1; col <= n_; ++col) {
                    heuristic_[index(row, col)] = 1.0 / std::max(due_time_[static_cast<std::size_t>(col - 1)], kEps);
                }
            }
        }
    }
    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) { pheromone_ = read_square_matrix(pheromone_array); }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) { heuristic_ = read_square_matrix(heuristic_array); }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_ + 1, n_ + 1); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_ + 1, n_ + 1); }
    double get_lowest_cost() const { return best_cost_; }
    py::tuple sample_trace() {
        auto sampled = sample_paths();
        return py::make_tuple(vector_to_numpy_generic(sampled.costs), sampled.trace);
    }
    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        auto paths = read_paths(paths_array);
        return vector_to_numpy_generic(gen_costs(paths));
    }
    void update_pheromone_from_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array, const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        auto paths = read_paths(paths_array);
        auto costs = read_costs(costs_array);
        update_pheromone(paths, costs);
    }
    double run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.costs);
        }
        return best_cost_;
    }
private:
    struct SampledBatch {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        BatchTrace trace;
    };
    std::vector<double> due_time_;
    std::vector<double> weights_;
    std::vector<double> processing_time_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    std::int64_t n_;
    double min_ = 0.1;
    double max_ = 1.0;
    double best_cost_ = std::numeric_limits<double>::infinity();
    std::vector<std::int64_t> best_sol_;
    std::mt19937_64 rng_;
    std::size_t index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * (n_ + 1) + col); }
    std::vector<std::int64_t> read_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 2 || buf.shape[0] != n_ || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (n, n_ants)");
        }
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + n_ * n_ants_);
    }
    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }
    SampledBatch sample_paths() {
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(n_ * n_ants_), 0);
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_ + 1), 1);
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1)] = 0;
        }
        std::vector<std::int64_t> prev(static_cast<std::size_t>(n_ants_), 0);
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<double> weights(static_cast<std::size_t>(n_ + 1), 0.0);
        for (std::int64_t step = 0; step < n_; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                double normalizer = 0.0;
                for (std::int64_t node = 0; node <= n_; ++node) {
                    const auto mask_idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(node);
                    if (visit_mask[mask_idx] == 0) {
                        weights[static_cast<std::size_t>(node)] = 0.0;
                        continue;
                    }
                    const double value = std::pow(std::max(pheromone_[index(prev[static_cast<std::size_t>(ant)], node)], kEps), alpha_) *
                                         std::pow(std::max(heuristic_[index(prev[static_cast<std::size_t>(ant)], node)], kEps), beta_);
                    weights[static_cast<std::size_t>(node)] = value;
                    normalizer += value;
                }
                const double threshold = unit_dist(rng_) * normalizer;
                double cumulative = 0.0;
                std::int64_t chosen = -1;
                std::int64_t last_feasible = -1;
                for (std::int64_t node = 0; node <= n_; ++node) {
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
                    chosen = last_feasible >= 0 ? last_feasible : 1;
                }
                paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] = chosen;
                prev[static_cast<std::size_t>(ant)] = chosen;
                visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ + 1) + static_cast<std::size_t>(chosen)] = 0;
            }
        }
        auto costs = gen_costs(paths_flat);
        BatchTrace trace;
        trace.paths_flat = paths_flat;
        trace.problem_size = n_;
        trace.n_ants = n_ants_;
        return SampledBatch{std::move(paths_flat), std::move(costs), std::move(trace)};
    }
    std::vector<double> gen_costs(const std::vector<std::int64_t>& paths_flat) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double total = 0.0;
            double completion = 0.0;
            for (std::int64_t step = 0; step < n_; ++step) {
                const auto job = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)] - 1;
                completion += processing_time_[static_cast<std::size_t>(job)];
                total += weights_[static_cast<std::size_t>(job)] * std::max(completion - due_time_[static_cast<std::size_t>(job)], 0.0);
            }
            costs[static_cast<std::size_t>(ant)] = total;
        }
        return costs;
    }
    void update_best(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        const auto best_iter = std::min_element(costs.begin(), costs.end());
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        if (*best_iter >= best_cost_) {
            return;
        }
        best_cost_ = *best_iter;
        best_sol_.resize(static_cast<std::size_t>(n_));
        for (std::int64_t step = 0; step < n_; ++step) {
            best_sol_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
    }
    void update_pheromone(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            deposit_path(paths_flat, best_idx, 1.0 / (std::max(*best_iter, 0.0) + 1.0));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, ant, 1.0 / (std::max(costs[static_cast<std::size_t>(ant)], 0.0) + 1.0));
            }
        }
        if (min_max_) {
            for (auto& value : pheromone_) {
                if (value > 1e-9 && value < min_) {
                    value = min_;
                }
                if (value > max_) {
                    value = max_;
                }
            }
        }
    }
    void deposit_path(const std::vector<std::int64_t>& paths_flat, int ant, double deposit) {
        if (n_ > 0) {
            const auto first = paths_flat[static_cast<std::size_t>(ant)];
            pheromone_[index(0, first)] += deposit;
        }
        for (std::int64_t step = 0; step + 1 < n_; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

class ACO_SOP {
public:
    ACO_SOP(
        const py::array_t<double, py::array::c_style | py::array::forcecast>& distances_array,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& prec_cons_array,
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
          prec_cons_(read_square_matrix(prec_cons_array)),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          elitist_(elitist),
          min_max_(min_max),
          n_(static_cast<std::int64_t>(distances_array.shape(0))),
          rng_(seed) {
        if (min_max_) {
            min_ = min_obj.is_none() ? 0.1 : py::cast<double>(min_obj);
        }
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(n_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(n_ * n_), min_max_ ? min_ : 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_));
        if (heuristic_.empty()) {
            heuristic_.resize(static_cast<std::size_t>(n_ * n_));
            for (std::int64_t row = 0; row < n_; ++row) {
                for (std::int64_t col = 0; col < n_; ++col) {
                    heuristic_[static_cast<std::size_t>(row * n_ + col)] = 1.0 / std::max(distances_[static_cast<std::size_t>(row * n_ + col)], kEps);
                }
            }
        }
    }
    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) { pheromone_ = read_square_matrix(pheromone_array); }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) { heuristic_ = read_square_matrix(heuristic_array); }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_, n_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_, n_); }
    double get_lowest_cost() const { return best_cost_; }
    py::tuple sample_trace() {
        auto sampled = sample_paths();
        return py::make_tuple(vector_to_numpy_generic(sampled.costs), sampled.trace);
    }
    py::array_t<double> gen_path_costs(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array) const {
        auto paths = read_paths(paths_array);
        return vector_to_numpy_generic(gen_costs(paths));
    }
    void update_pheromone_from_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& paths_array, const py::array_t<double, py::array::c_style | py::array::forcecast>& costs_array) {
        auto paths = read_paths(paths_array);
        auto costs = read_costs(costs_array);
        update_pheromone(paths, costs);
    }
    double run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto sampled = sample_paths();
            update_best(sampled.paths_flat, sampled.costs);
            update_pheromone(sampled.paths_flat, sampled.costs);
        }
        return best_cost_;
    }
private:
    struct SampledBatch {
        std::vector<std::int64_t> paths_flat;
        std::vector<double> costs;
        BatchTrace trace;
    };
    std::vector<double> distances_;
    std::vector<double> prec_cons_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    bool elitist_;
    bool min_max_;
    std::int64_t n_;
    double min_ = 0.1;
    double max_ = -1.0;
    double best_cost_ = std::numeric_limits<double>::infinity();
    std::vector<std::int64_t> best_path_;
    std::mt19937_64 rng_;
    std::size_t index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * n_ + col); }
    std::vector<std::int64_t> read_paths(const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 2 || buf.shape[0] != n_ || buf.shape[1] != n_ants_) {
            throw std::invalid_argument("paths must have shape (n, n_ants)");
        }
        const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
        return std::vector<std::int64_t>(ptr, ptr + n_ * n_ants_);
    }
    std::vector<double> read_costs(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const {
        auto buf = array.request();
        if (buf.ndim != 1 || buf.shape[0] != n_ants_) {
            throw std::invalid_argument("costs must have shape (n_ants,)");
        }
        const auto* ptr = static_cast<const double*>(buf.ptr);
        return std::vector<double>(ptr, ptr + n_ants_);
    }
    void update_prec(std::vector<double>& prec, const std::vector<std::int64_t>& actions) const {
        for (int ant = 0; ant < n_ants_; ++ant) {
            const auto action = actions[static_cast<std::size_t>(ant)];
            for (std::int64_t row = 0; row < n_; ++row) {
                prec[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ * n_) + static_cast<std::size_t>(row * n_ + action)] = 0.0;
            }
        }
    }
    SampledBatch sample_paths() {
        std::vector<std::int64_t> paths_flat(static_cast<std::size_t>(n_ * n_ants_), 0);
        std::vector<double> prec(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_ * n_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::copy(prec_cons_.begin(), prec_cons_.end(), prec.begin() + static_cast<std::ptrdiff_t>(ant * n_ * n_));
        }
        std::vector<unsigned char> visit_mask(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 1);
        std::vector<std::int64_t> prev(static_cast<std::size_t>(n_ants_), 0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_)] = 0;
        }
        update_prec(prec, prev);
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<double> weights(static_cast<std::size_t>(n_), 0.0);
        for (std::int64_t step = 1; step < n_; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                double normalizer = 0.0;
                for (std::int64_t node = 0; node < n_; ++node) {
                    bool prec_ok = true;
                    for (std::int64_t pred = 0; pred < n_; ++pred) {
                        if (prec[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_ * n_) + static_cast<std::size_t>(node * n_ + pred)] != 0.0) {
                            prec_ok = false;
                            break;
                        }
                    }
                    const auto mask_idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node);
                    if (visit_mask[mask_idx] == 0 || !prec_ok) {
                        weights[static_cast<std::size_t>(node)] = 0.0;
                        continue;
                    }
                    const double value = std::pow(std::max(pheromone_[index(prev[static_cast<std::size_t>(ant)], node)], kEps), alpha_) *
                                         std::pow(std::max(heuristic_[index(prev[static_cast<std::size_t>(ant)], node)], kEps), beta_);
                    weights[static_cast<std::size_t>(node)] = value;
                    normalizer += value;
                }
                const double threshold = unit_dist(rng_) * normalizer;
                double cumulative = 0.0;
                std::int64_t chosen = -1;
                std::int64_t last_feasible = -1;
                for (std::int64_t node = 0; node < n_; ++node) {
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
                prev[static_cast<std::size_t>(ant)] = chosen;
            }
            update_prec(prec, prev);
            for (int ant = 0; ant < n_ants_; ++ant) {
                visit_mask[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(prev[static_cast<std::size_t>(ant)])] = 0;
            }
        }
        auto costs = gen_costs(paths_flat);
        BatchTrace trace;
        trace.paths_flat = paths_flat;
        trace.problem_size = n_;
        trace.n_ants = n_ants_;
        return SampledBatch{std::move(paths_flat), std::move(costs), std::move(trace)};
    }
    std::vector<double> gen_costs(const std::vector<std::int64_t>& paths_flat) const {
        std::vector<double> costs(static_cast<std::size_t>(n_ants_), 0.0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            double total = 0.0;
            for (std::int64_t step = 0; step + 1 < n_; ++step) {
                const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
                const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
                total += distances_[index(curr, next)];
            }
            costs[static_cast<std::size_t>(ant)] = total;
        }
        return costs;
    }
    void update_best(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        const auto best_iter = std::min_element(costs.begin(), costs.end());
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        if (*best_iter >= best_cost_) {
            return;
        }
        best_cost_ = *best_iter;
        best_path_.resize(static_cast<std::size_t>(n_));
        for (std::int64_t step = 0; step < n_; ++step) {
            best_path_[static_cast<std::size_t>(step)] = paths_flat[static_cast<std::size_t>(step * n_ants_ + best_idx)];
        }
        if (min_max_) {
            const double next_max = static_cast<double>(n_) / std::max(best_cost_, kEps);
            if (max_ < 0.0) {
                const double current_max = *std::max_element(pheromone_.begin(), pheromone_.end());
                for (auto& value : pheromone_) {
                    value *= next_max / std::max(current_max, kEps);
                }
            }
            max_ = next_max;
        }
    }
    void update_pheromone(const std::vector<std::int64_t>& paths_flat, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            deposit_path(paths_flat, best_idx, 1.0 / std::max(*best_iter, kEps));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_path(paths_flat, ant, 1.0 / std::max(costs[static_cast<std::size_t>(ant)], kEps));
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
    }
    void deposit_path(const std::vector<std::int64_t>& paths_flat, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < n_; ++step) {
            const auto curr = paths_flat[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = paths_flat[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

class ACO_RCPSP_CONSTRUCT {
public:
    ACO_RCPSP_CONSTRUCT(
        std::int64_t n,
        const std::vector<std::int64_t>& indegrees,
        const std::vector<std::vector<std::int64_t>>& adjlist,
        const std::vector<std::int64_t>& durations,
        const std::vector<std::vector<std::int64_t>>& resource_requirements,
        const std::vector<std::int64_t>& capacities,
        const std::vector<std::int64_t>& earliest_starts,
        const std::vector<std::int64_t>& latest_starts,
        int n_ants = 5,
        double decay = 0.975,
        double alpha = 1.0,
        double beta = 2.0,
        double gamma = 0.0,
        double c = 0.6,
        double q = 1.0,
        bool elitist = false,
        bool min_max = false,
        double min = 0.1,
        py::object pheromone_obj = py::none(),
        py::object heuristic_obj = py::none(),
        std::uint64_t seed = std::random_device{}())
        : n_(n),
          indegrees_(indegrees),
          adjlist_(adjlist),
          durations_(durations),
          resource_requirements_(resource_requirements),
          capacities_(capacities),
          earliest_starts_(earliest_starts),
          latest_starts_(latest_starts),
          n_ants_(n_ants),
          decay_(decay),
          alpha_(alpha),
          beta_(beta),
          gamma_(gamma),
          c_(c),
          q_(q),
          elitist_(elitist),
          min_max_(min_max),
          min_(min),
          rng_(seed) {
        if (static_cast<std::int64_t>(indegrees_.size()) != n_) {
            throw std::invalid_argument("indegrees has incorrect shape");
        }
        if (static_cast<std::int64_t>(durations_.size()) != n_) {
            throw std::invalid_argument("durations has incorrect shape");
        }
        if (static_cast<std::int64_t>(resource_requirements_.size()) != n_) {
            throw std::invalid_argument("resource requirements has incorrect shape");
        }
        if (static_cast<std::int64_t>(earliest_starts_.size()) != n_ || static_cast<std::int64_t>(latest_starts_.size()) != n_) {
            throw std::invalid_argument("time-window vectors have incorrect shape");
        }
        for (const auto& requirements : resource_requirements_) {
            if (requirements.size() != capacities_.size()) {
                throw std::invalid_argument("resource requirements must match capacities");
            }
        }
        predlist_.assign(static_cast<std::size_t>(n_), {});
        for (std::int64_t src = 0; src < n_; ++src) {
            for (const auto dst : adjlist_[static_cast<std::size_t>(src)]) {
                predlist_[static_cast<std::size_t>(dst)].push_back(src);
            }
        }
        pheromone_ = read_optional_square_matrix(pheromone_obj, static_cast<std::size_t>(n_));
        if (pheromone_.empty()) {
            pheromone_.assign(static_cast<std::size_t>(n_ * n_), min_max_ ? min_ : 1.0);
        }
        heuristic_ = read_optional_square_matrix(heuristic_obj, static_cast<std::size_t>(n_));
        if (heuristic_.empty()) {
            heuristic_ = build_default_heuristic();
        }
    }
    void set_seed(std::uint64_t seed) { rng_.seed(seed); }
    void set_pheromone(const py::array_t<double, py::array::c_style | py::array::forcecast>& pheromone_array) { pheromone_ = read_square_matrix(pheromone_array); }
    void set_heuristic(const py::array_t<double, py::array::c_style | py::array::forcecast>& heuristic_array) { heuristic_ = read_square_matrix(heuristic_array); }
    py::array_t<double> get_pheromone() const { return matrix_to_numpy_generic(pheromone_, n_, n_); }
    py::array_t<double> get_heuristic() const { return matrix_to_numpy_generic(heuristic_, n_, n_); }
    py::tuple get_best_solution() const {
        return py::make_tuple(static_cast<double>(best_cost_), vector_to_numpy_int64(best_route_), vector_to_numpy_int64(best_schedule_));
    }
    BatchTrace sample_trace() {
        BatchTrace trace;
        trace.problem_size = n_;
        trace.n_ants = n_ants_;
        trace.paths_flat = sample_routes();
        return trace;
    }
    py::tuple sample_trace_costs() {
        auto routes = sample_routes();
        auto evaluated = evaluate_routes(routes);
        update_best(routes, evaluated.costs, evaluated.schedules);
        BatchTrace trace;
        trace.problem_size = n_;
        trace.n_ants = n_ants_;
        trace.paths_flat = routes;
        return py::make_tuple(
            vector_to_numpy_generic(evaluated.costs),
            matrix_to_numpy_int64(evaluated.schedules, n_ants_, n_),
            trace);
    }
    py::tuple run(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            auto routes = sample_routes();
            auto evaluated = evaluate_routes(routes);
            update_best(routes, evaluated.costs, evaluated.schedules);
            update_pheromone(routes, evaluated.costs);
        }
        return get_best_solution();
    }
private:
    struct ResourceState {
        explicit ResourceState(std::int64_t capacity) : capacity(capacity), available(capacity) {}

        std::int64_t available_timestamp(std::int64_t amount) const {
            if (amount == 0) {
                return 0;
            }
            std::int64_t remaining = amount - available;
            if (remaining <= 0) {
                return last_event_time;
            }
            for (const auto& release : queue) {
                remaining -= release.second;
                if (remaining <= 0) {
                    return release.first;
                }
            }
            throw std::runtime_error("unable to find a feasible resource timestamp");
        }

        void request(std::int64_t timestamp, std::int64_t amount, std::int64_t duration) {
            if (timestamp < last_event_time) {
                throw std::runtime_error("resource requests must be processed in chronological order");
            }
            last_event_time = timestamp;
            std::vector<std::pair<std::int64_t, std::int64_t>> pending;
            pending.reserve(queue.size() + 1);
            for (const auto& release : queue) {
                if (release.first <= timestamp) {
                    available += release.second;
                } else {
                    pending.push_back(release);
                }
            }
            pending.emplace_back(timestamp + duration, amount);
            std::sort(pending.begin(), pending.end());
            queue = std::move(pending);
            available -= amount;
            if (available < 0) {
                throw std::runtime_error("resource capacity underflow");
            }
        }

        std::int64_t capacity = 0;
        std::int64_t available = 0;
        std::int64_t last_event_time = 0;
        std::vector<std::pair<std::int64_t, std::int64_t>> queue;
    };

    struct EvaluatedRoutes {
        std::vector<double> costs;
        std::vector<std::int64_t> schedules;
    };

    std::int64_t n_;
    std::vector<std::int64_t> indegrees_;
    std::vector<std::vector<std::int64_t>> adjlist_;
    std::vector<std::vector<std::int64_t>> predlist_;
    std::vector<std::int64_t> durations_;
    std::vector<std::vector<std::int64_t>> resource_requirements_;
    std::vector<std::int64_t> capacities_;
    std::vector<std::int64_t> earliest_starts_;
    std::vector<std::int64_t> latest_starts_;
    std::vector<double> pheromone_;
    std::vector<double> heuristic_;
    int n_ants_;
    double decay_;
    double alpha_;
    double beta_;
    double gamma_;
    double c_;
    double q_;
    bool elitist_;
    bool min_max_;
    double min_;
    double max_ = std::numeric_limits<double>::infinity();
    std::int64_t best_cost_ = std::numeric_limits<std::int64_t>::max();
    std::vector<std::int64_t> best_route_;
    std::vector<std::int64_t> best_schedule_;
    std::mt19937_64 rng_;
    std::size_t index(std::int64_t row, std::int64_t col) const { return static_cast<std::size_t>(row * n_ + col); }
    std::vector<double> build_default_heuristic() const {
        std::vector<double> wrup(static_cast<std::size_t>(n_), 0.0);
        double wrup_min = std::numeric_limits<double>::infinity();
        double wrup_max = -std::numeric_limits<double>::infinity();
        constexpr double omega = 0.3;
        for (std::int64_t node = 0; node < n_; ++node) {
            double value = omega * static_cast<double>(adjlist_[static_cast<std::size_t>(node)].size());
            const auto& requirements = resource_requirements_[static_cast<std::size_t>(node)];
            for (std::size_t resource_idx = 0; resource_idx < requirements.size(); ++resource_idx) {
                value += (1.0 - omega) * static_cast<double>(requirements[resource_idx]) /
                         std::max(static_cast<double>(capacities_[resource_idx]), kEps);
            }
            wrup[static_cast<std::size_t>(node)] = value;
            wrup_min = std::min(wrup_min, value);
            wrup_max = std::max(wrup_max, value);
        }
        for (auto& value : wrup) {
            value = value - wrup_min + 1.0;
        }
        const double wrup_scale = *std::max_element(wrup.begin(), wrup.end());

        std::vector<std::int64_t> succ_closure_sizes(static_cast<std::size_t>(n_), 0);
        std::vector<unsigned char> closure(static_cast<std::size_t>(n_ * n_), 0);
        for (std::int64_t node = n_ - 1; node >= 0; --node) {
            for (const auto succ : adjlist_[static_cast<std::size_t>(node)]) {
                closure[index(node, succ)] = 1;
                for (std::int64_t other = 0; other < n_; ++other) {
                    if (closure[index(succ, other)] != 0) {
                        closure[index(node, other)] = 1;
                    }
                }
            }
            std::int64_t count = 0;
            for (std::int64_t other = 0; other < n_; ++other) {
                if (closure[index(node, other)] != 0) {
                    ++count;
                }
            }
            succ_closure_sizes[static_cast<std::size_t>(node)] = count;
        }
        const auto min_succ = *std::min_element(succ_closure_sizes.begin(), succ_closure_sizes.end());

        std::vector<double> heuristic(static_cast<std::size_t>(n_ * n_), 1.0);
        for (std::int64_t col = 0; col < n_; ++col) {
            const double grpwa = static_cast<double>(succ_closure_sizes[static_cast<std::size_t>(col)] - min_succ + 1);
            const double combined = (wrup[static_cast<std::size_t>(col)] / std::max(wrup_scale, kEps)) * grpwa;
            for (std::int64_t row = 0; row < n_; ++row) {
                heuristic[index(row, col)] = combined;
            }
        }
        return heuristic;
    }
    std::vector<std::int64_t> sample_routes() {
        std::vector<double> probmat(pheromone_.size(), 0.0);
        for (std::size_t i = 0; i < probmat.size(); ++i) {
            probmat[i] = std::pow(std::max(pheromone_[i], kEps), alpha_) * std::pow(std::max(heuristic_[i], kEps), beta_);
        }
        std::vector<std::int64_t> routes(static_cast<std::size_t>(n_ * n_ants_), 0);
        std::vector<unsigned char> not_visited(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 1);
        std::vector<std::int64_t> indegrees(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::copy(indegrees_.begin(), indegrees_.end(), indegrees.begin() + static_cast<std::ptrdiff_t>(ant * n_));
            routes[static_cast<std::size_t>(ant)] = 0;
        }
        std::vector<std::int64_t> prev(static_cast<std::size_t>(n_ants_), 0);
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
        std::vector<double> weights(static_cast<std::size_t>(n_), 0.0);
        for (std::int64_t step = 0; step < n_ - 1; ++step) {
            for (int ant = 0; ant < n_ants_; ++ant) {
                not_visited[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(prev[static_cast<std::size_t>(ant)])] = 0;
                for (const auto succ : adjlist_[static_cast<std::size_t>(prev[static_cast<std::size_t>(ant)])]) {
                    indegrees[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(succ)] -= 1;
                }
            }
            for (int ant = 0; ant < n_ants_; ++ant) {
                double normalizer = 0.0;
                for (std::int64_t node = 0; node < n_; ++node) {
                    const auto idx = static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node);
                    if (not_visited[idx] == 0 || indegrees[idx] != 0) {
                        weights[static_cast<std::size_t>(node)] = 0.0;
                        continue;
                    }
                    double value = 0.0;
                    if (gamma_ < 0.05 || c_ == 1.0) {
                        value = probmat[index(prev[static_cast<std::size_t>(ant)], node)];
                    } else {
                        double summed = 0.0;
                        for (std::int64_t hist = 0; hist <= step; ++hist) {
                            double factor = 1.0;
                            if (gamma_ != 1.0) {
                                factor = std::pow(gamma_, static_cast<double>(step - hist));
                            }
                            summed += pheromone_[index(routes[static_cast<std::size_t>(hist * n_ants_ + ant)], node)] * factor;
                        }
                        const double summation = std::pow(std::max(summed, kEps), alpha_) * std::pow(std::max(heuristic_[index(prev[static_cast<std::size_t>(ant)], node)], kEps), beta_);
                        if (c_ == 0.0) {
                            value = summation;
                        } else {
                            const double direct = probmat[index(prev[static_cast<std::size_t>(ant)], node)];
                            value = c_ * direct + (1.0 - c_) * summation;
                        }
                    }
                    weights[static_cast<std::size_t>(node)] = value;
                    normalizer += value;
                }
                const double threshold = unit_dist(rng_) * normalizer;
                double cumulative = 0.0;
                std::int64_t chosen = -1;
                std::int64_t last_feasible = -1;
                for (std::int64_t node = 0; node < n_; ++node) {
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
                routes[static_cast<std::size_t>((step + 1) * n_ants_ + ant)] = chosen;
                prev[static_cast<std::size_t>(ant)] = chosen;
            }
        }
        return routes;
    }

    EvaluatedRoutes evaluate_routes(const std::vector<std::int64_t>& routes) const {
        EvaluatedRoutes evaluated;
        evaluated.costs.assign(static_cast<std::size_t>(n_ants_), 0.0);
        evaluated.schedules.assign(static_cast<std::size_t>(n_ants_) * static_cast<std::size_t>(n_), 0);
        for (int ant = 0; ant < n_ants_; ++ant) {
            std::vector<std::int64_t> end_times(static_cast<std::size_t>(n_), 0);
            std::vector<ResourceState> resources;
            resources.reserve(capacities_.size());
            for (const auto capacity : capacities_) {
                resources.emplace_back(capacity);
            }
            for (std::int64_t step = 0; step < n_; ++step) {
                const auto node = routes[static_cast<std::size_t>(step * n_ants_ + ant)];
                std::int64_t earliest = earliest_starts_[static_cast<std::size_t>(node)];
                for (const auto pred : predlist_[static_cast<std::size_t>(node)]) {
                    earliest = std::max(earliest, end_times[static_cast<std::size_t>(pred)]);
                }
                std::int64_t arrange = 0;
                const auto& requirements = resource_requirements_[static_cast<std::size_t>(node)];
                for (std::size_t resource_idx = 0; resource_idx < requirements.size(); ++resource_idx) {
                    const auto amount = requirements[resource_idx];
                    if (amount > 0) {
                        arrange = std::max(arrange, resources[resource_idx].available_timestamp(amount));
                    }
                }
                arrange = std::min(std::max(arrange, earliest), latest_starts_[static_cast<std::size_t>(node)]);
                for (std::size_t resource_idx = 0; resource_idx < requirements.size(); ++resource_idx) {
                    const auto amount = requirements[resource_idx];
                    if (amount > 0) {
                        resources[resource_idx].request(arrange, amount, durations_[static_cast<std::size_t>(node)]);
                    }
                }
                evaluated.schedules[static_cast<std::size_t>(ant) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(node)] = arrange;
                end_times[static_cast<std::size_t>(node)] = arrange + durations_[static_cast<std::size_t>(node)];
            }
            evaluated.costs[static_cast<std::size_t>(ant)] = static_cast<double>(end_times.back());
        }
        return evaluated;
    }

    void update_best(
        const std::vector<std::int64_t>& routes,
        const std::vector<double>& costs,
        const std::vector<std::int64_t>& schedules) {
        const auto best_iter = std::min_element(costs.begin(), costs.end());
        const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
        const auto best_cost = static_cast<std::int64_t>(std::llround(*best_iter));
        if (best_cost >= best_cost_) {
            return;
        }
        best_cost_ = best_cost;
        best_route_.resize(static_cast<std::size_t>(n_));
        best_schedule_.resize(static_cast<std::size_t>(n_));
        for (std::int64_t step = 0; step < n_; ++step) {
            best_route_[static_cast<std::size_t>(step)] = routes[static_cast<std::size_t>(step * n_ants_ + best_idx)];
            best_schedule_[static_cast<std::size_t>(step)] = schedules[static_cast<std::size_t>(best_idx) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(step)];
        }
        max_ = q_ * static_cast<double>(n_) / std::max(*best_iter, 1.0);
    }

    void update_pheromone(const std::vector<std::int64_t>& routes, const std::vector<double>& costs) {
        for (auto& value : pheromone_) {
            value *= decay_;
        }
        if (!best_route_.empty()) {
            deposit_route(best_route_, q_ / std::max(static_cast<double>(best_cost_), 1.0));
        }
        if (elitist_) {
            const auto best_iter = std::min_element(costs.begin(), costs.end());
            const auto best_idx = static_cast<int>(std::distance(costs.begin(), best_iter));
            deposit_ant(routes, best_idx, q_ / std::max(*best_iter, 1.0));
        } else {
            for (int ant = 0; ant < n_ants_; ++ant) {
                deposit_ant(routes, ant, q_ / std::max(costs[static_cast<std::size_t>(ant)], 1.0));
            }
        }
        if (min_max_) {
            for (auto& value : pheromone_) {
                if (value > max_) {
                    value = max_;
                }
                if (value < min_) {
                    value = min_;
                }
            }
        }
    }

    void deposit_route(const std::vector<std::int64_t>& route, double deposit) {
        for (std::int64_t step = 0; step + 1 < n_; ++step) {
            pheromone_[index(route[static_cast<std::size_t>(step)], route[static_cast<std::size_t>(step + 1)])] += deposit;
        }
    }

    void deposit_ant(const std::vector<std::int64_t>& routes, int ant, double deposit) {
        for (std::int64_t step = 0; step + 1 < n_; ++step) {
            const auto curr = routes[static_cast<std::size_t>(step * n_ants_ + ant)];
            const auto next = routes[static_cast<std::size_t>((step + 1) * n_ants_ + ant)];
            pheromone_[index(curr, next)] += deposit;
        }
    }
};

}  // namespace

inline void bind_additional_aco_classes(py::module_& m) {
    py::class_<ACO_BPP>(m, "ACO_BPP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, int, double, double, double, bool, py::object, py::object, double, std::uint64_t>(),
             py::arg("demand"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("capacity") = 150.0,
             py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_BPP::set_seed)
        .def("set_pheromone", &ACO_BPP::set_pheromone)
        .def("set_heuristic", &ACO_BPP::set_heuristic)
        .def("get_pheromone", &ACO_BPP::get_pheromone)
        .def("get_heuristic", &ACO_BPP::get_heuristic)
        .def("get_demand", &ACO_BPP::get_demand)
        .def("get_shortest_path", &ACO_BPP::get_shortest_path)
        .def("get_best_fitness", &ACO_BPP::get_best_fitness)
        .def("get_lowest_cost", &ACO_BPP::get_lowest_cost)
        .def("sample_trace", &ACO_BPP::sample_trace)
        .def("gen_path_costs", &ACO_BPP::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_BPP::update_pheromone_from_paths)
        .def("run", &ACO_BPP::run);

    py::class_<ACO_MKP>(m, "ACO_MKP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, int, double, double, double, bool, bool, py::object, py::object, py::object, std::uint64_t>(),
             py::arg("prize"), py::arg("weight"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("min_max") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("min") = py::none(),
             py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_MKP::set_seed)
        .def("set_pheromone", &ACO_MKP::set_pheromone)
        .def("set_heuristic", &ACO_MKP::set_heuristic)
        .def("get_pheromone", &ACO_MKP::get_pheromone)
        .def("get_heuristic", &ACO_MKP::get_heuristic)
        .def("get_lowest_cost", &ACO_MKP::get_lowest_cost)
        .def("sample_trace", &ACO_MKP::sample_trace)
        .def("gen_sol_obj", &ACO_MKP::gen_sol_obj)
        .def("update_pheromone_from_paths", &ACO_MKP::update_pheromone_from_paths)
        .def("run", &ACO_MKP::run);

    py::class_<ACO_OP>(m, "ACO_OP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, double, int, double, double, double, bool, bool, py::object, py::object, py::object, py::object, std::uint64_t>(),
             py::arg("distances"), py::arg("prizes"), py::arg("max_len"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("min_max") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("min") = py::none(), py::arg("k_sparse") = py::none(), py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_OP::set_seed)
        .def("set_pheromone", &ACO_OP::set_pheromone)
        .def("set_heuristic", &ACO_OP::set_heuristic)
        .def("get_pheromone", &ACO_OP::get_pheromone)
        .def("get_heuristic", &ACO_OP::get_heuristic)
        .def("get_lowest_cost", &ACO_OP::get_lowest_cost)
        .def("sample_trace", &ACO_OP::sample_trace)
        .def("gen_sol_obj", &ACO_OP::gen_sol_obj)
        .def("update_pheromone_from_paths", &ACO_OP::update_pheromone_from_paths)
        .def("run", &ACO_OP::run);

    py::class_<ACO_PCTSP>(m, "ACO_PCTSP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, int, double, double, double, bool, bool, py::object, py::object, py::object, std::uint64_t>(),
             py::arg("distances"), py::arg("prizes"), py::arg("penalties"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("min_max") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("min") = py::none(), py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_PCTSP::set_seed)
        .def("set_pheromone", &ACO_PCTSP::set_pheromone)
        .def("set_heuristic", &ACO_PCTSP::set_heuristic)
        .def("get_pheromone", &ACO_PCTSP::get_pheromone)
        .def("get_heuristic", &ACO_PCTSP::get_heuristic)
        .def("get_lowest_cost", &ACO_PCTSP::get_lowest_cost)
        .def("sample_trace", &ACO_PCTSP::sample_trace)
        .def("gen_sol_obj", &ACO_PCTSP::gen_sol_obj)
        .def("update_pheromone_from_paths", &ACO_PCTSP::update_pheromone_from_paths)
        .def("run", &ACO_PCTSP::run);

    py::class_<ACO_SMTWTP>(m, "ACO_SMTWTP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, int, double, double, double, bool, bool, py::object, py::object, py::object, std::uint64_t>(),
             py::arg("due_time"), py::arg("weights"), py::arg("processing_time"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("min_max") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("min") = py::none(), py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_SMTWTP::set_seed)
        .def("set_pheromone", &ACO_SMTWTP::set_pheromone)
        .def("set_heuristic", &ACO_SMTWTP::set_heuristic)
        .def("get_pheromone", &ACO_SMTWTP::get_pheromone)
        .def("get_heuristic", &ACO_SMTWTP::get_heuristic)
        .def("get_lowest_cost", &ACO_SMTWTP::get_lowest_cost)
        .def("sample_trace", &ACO_SMTWTP::sample_trace)
        .def("gen_path_costs", &ACO_SMTWTP::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_SMTWTP::update_pheromone_from_paths)
        .def("run", &ACO_SMTWTP::run);

    py::class_<ACO_SOP>(m, "ACO_SOP")
        .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const py::array_t<double, py::array::c_style | py::array::forcecast>&, int, double, double, double, bool, bool, py::object, py::object, py::object, std::uint64_t>(),
             py::arg("distances"), py::arg("prec_cons"), py::arg("n_ants") = 20, py::arg("decay") = 0.9, py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("elitist") = false, py::arg("min_max") = false, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("min") = py::none(), py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_SOP::set_seed)
        .def("set_pheromone", &ACO_SOP::set_pheromone)
        .def("set_heuristic", &ACO_SOP::set_heuristic)
        .def("get_pheromone", &ACO_SOP::get_pheromone)
        .def("get_heuristic", &ACO_SOP::get_heuristic)
        .def("get_lowest_cost", &ACO_SOP::get_lowest_cost)
        .def("sample_trace", &ACO_SOP::sample_trace)
        .def("gen_path_costs", &ACO_SOP::gen_path_costs)
        .def("update_pheromone_from_paths", &ACO_SOP::update_pheromone_from_paths)
        .def("run", &ACO_SOP::run);

    py::class_<ACO_RCPSP_CONSTRUCT>(m, "ACO_RCPSP_CONSTRUCT")
           .def(py::init<std::int64_t, const std::vector<std::int64_t>&, const std::vector<std::vector<std::int64_t>>&, const std::vector<std::int64_t>&, const std::vector<std::vector<std::int64_t>>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, int, double, double, double, double, double, double, bool, bool, double, py::object, py::object, std::uint64_t>(),
               py::arg("n"), py::arg("indegrees"), py::arg("adjlist"), py::arg("durations"), py::arg("resource_requirements"), py::arg("capacities"), py::arg("earliest_starts"), py::arg("latest_starts"),
               py::arg("n_ants") = 5, py::arg("decay") = 0.975, py::arg("alpha") = 1.0, py::arg("beta") = 2.0, py::arg("gamma") = 0.0, py::arg("c") = 0.6, py::arg("q") = 1.0,
               py::arg("elitist") = false, py::arg("min_max") = false, py::arg("min") = 0.1, py::arg("pheromone") = py::none(), py::arg("heuristic") = py::none(), py::arg("seed") = std::random_device{}())
        .def("set_seed", &ACO_RCPSP_CONSTRUCT::set_seed)
        .def("set_pheromone", &ACO_RCPSP_CONSTRUCT::set_pheromone)
        .def("set_heuristic", &ACO_RCPSP_CONSTRUCT::set_heuristic)
        .def("get_pheromone", &ACO_RCPSP_CONSTRUCT::get_pheromone)
        .def("get_heuristic", &ACO_RCPSP_CONSTRUCT::get_heuristic)
           .def("get_best_solution", &ACO_RCPSP_CONSTRUCT::get_best_solution)
           .def("sample_trace", &ACO_RCPSP_CONSTRUCT::sample_trace)
           .def("sample_trace_costs", &ACO_RCPSP_CONSTRUCT::sample_trace_costs)
           .def("run", &ACO_RCPSP_CONSTRUCT::run);
}
