#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct SemanticParse {
    bool ok;
    std::string priority_key;
    std::string remainder;
};

static std::string to_lower(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

static SemanticParse parse_semantic(const std::string &strategy) {
    std::string s = strategy;
    std::string sl = to_lower(s);
    if (sl.rfind("semantic", 0) != 0) {
        return {false, "", ""};
    }

    std::string rest = s.substr(8);
    while (!rest.empty() && rest[0] == '_') rest.erase(0, 1);
    if (rest.empty()) return {false, "", ""};

    std::string group_part;
    std::string remainder;
    size_t pos = rest.find('_');
    if (pos != std::string::npos) {
        group_part = rest.substr(0, pos);
        remainder = rest.substr(pos + 1);
    } else {
        group_part = rest;
        remainder = "random";
    }

    group_part = to_lower(group_part);
    remainder = to_lower(remainder);

    std::string key;
    if (group_part == "v" || group_part == "join" || group_part == "j") {
        key = "V";
    } else if (group_part == "pred" || group_part == "p") {
        key = "PRED";
    } else if (group_part == "slack" || group_part == "s") {
        key = "SLACK";
    } else {
        return {false, "", ""};
    }

    if (remainder != "random" && remainder != "greedy") {
        remainder = "random";
    }

    return {true, key, remainder};
}

static int64_t edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
}

static void build_logical_adj(
    int n_vars,
    const int *quad_u,
    const int *quad_v,
    int m,
    std::vector<std::vector<int>> &logical_adj,
    std::vector<int> &log_deg
) {
    logical_adj.assign(n_vars, {});
    log_deg.assign(n_vars, 0);
    logical_adj.reserve(n_vars);
    for (int i = 0; i < m; ++i) {
        int u = quad_u[i];
        int v = quad_v[i];
        if (u < 0 || v < 0 || u >= n_vars || v >= n_vars || u == v) {
            continue;
        }
        logical_adj[u].push_back(v);
        logical_adj[v].push_back(u);
        log_deg[u] += 1;
        log_deg[v] += 1;
    }
}

static bool has_neighbor(const std::vector<std::vector<int>> &adj, int qpos, int neighbor_pos) {
    const auto &neighbors = adj[qpos];
    return std::binary_search(neighbors.begin(), neighbors.end(), neighbor_pos);
}

static void build_weight_adj(
    int n_vars,
    const int *quad_u,
    const int *quad_v,
    const double *quad_w,
    int m,
    std::vector<std::unordered_map<int, double>> &weight_adj,
    std::vector<double> &abs_weight_sum
) {
    weight_adj.assign(n_vars, {});
    abs_weight_sum.assign(n_vars, 0.0);
    for (int i = 0; i < m; ++i) {
        int u = quad_u[i];
        int v = quad_v[i];
        if (u < 0 || v < 0 || u >= n_vars || v >= n_vars || u == v) {
            continue;
        }
        double w = quad_w[i];
        if (w < 0) w = -w;
        weight_adj[u][v] = w;
        weight_adj[v][u] = w;
        abs_weight_sum[u] += w;
        abs_weight_sum[v] += w;
    }
}

static std::vector<int> order_by_degree_desc(const std::vector<int> &deg) {
    std::vector<int> order(deg.size());
    for (size_t i = 0; i < deg.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&deg](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] > deg[b];
        return a < b;
    });
    return order;
}

static bool build_mapping_random(
    int n_vars,
    const std::vector<int> &qubits_order_pos,
    std::vector<int> &mapping_pos,
    std::mt19937 &rng
) {
    if (static_cast<int>(qubits_order_pos.size()) < n_vars) return false;
    std::vector<int> shuffled = qubits_order_pos;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    mapping_pos.assign(n_vars, -1);
    for (int i = 0; i < n_vars; ++i) {
        mapping_pos[i] = shuffled[i];
    }
    return true;
}

static bool build_mapping_greedy(
    int n_vars,
    const std::vector<std::vector<int>> &logical_adj,
    const std::vector<int> &log_deg,
    const std::vector<std::vector<int>> &phys_adj,
    const std::vector<int> &qubits_order_pos,
    std::vector<int> &mapping_pos
) {
    if (static_cast<int>(qubits_order_pos.size()) < n_vars) return false;

    std::vector<int> vars_order(n_vars);
    for (int i = 0; i < n_vars; ++i) vars_order[i] = i;
    std::sort(vars_order.begin(), vars_order.end(), [&log_deg](int a, int b) {
        if (log_deg[a] != log_deg[b]) return log_deg[a] > log_deg[b];
        return a < b;
    });

    mapping_pos.assign(n_vars, -1);
    std::vector<char> used(qubits_order_pos.size(), 0);

    auto score_place = [&](int v, int qpos) -> int {
        int s = 0;
        for (int u : logical_adj[v]) {
            int mapped = mapping_pos[u];
            if (mapped < 0) continue;
            if (has_neighbor(phys_adj, qpos, mapped)) {
                s += 1;
            }
        }
        return s;
    };

    for (int v : vars_order) {
        int best_qpos = -1;
        int best_s = -1;
        for (size_t qi = 0; qi < qubits_order_pos.size(); ++qi) {
            int qpos = qubits_order_pos[qi];
            if (used[qpos]) continue;
            int s = score_place(v, qpos);
            if (s > best_s) {
                best_s = s;
                best_qpos = qpos;
            }
        }
        if (best_qpos < 0) {
            for (size_t qi = 0; qi < qubits_order_pos.size(); ++qi) {
                int qpos = qubits_order_pos[qi];
                if (!used[qpos]) {
                    best_qpos = qpos;
                    break;
                }
            }
        }
        if (best_qpos < 0) return false;
        mapping_pos[v] = best_qpos;
        used[best_qpos] = 1;
    }
    return true;
}

static bool build_mapping_seeded_neighbor_greedy(
    int n_vars,
    const std::vector<std::vector<int>> &phys_adj,
    const std::vector<int> &phys_deg,
    const std::vector<int> &qubits_ids,
    const std::vector<std::unordered_map<int, double>> &weight_adj,
    const std::vector<double> &abs_weight_sum,
    std::vector<int> &mapping_pos
) {
    if (static_cast<int>(phys_adj.size()) < n_vars) return false;

    int seed_var = 0;
    double best_sum = -1.0;
    for (int v = 0; v < n_vars; ++v) {
        if (abs_weight_sum[v] > best_sum) {
            best_sum = abs_weight_sum[v];
            seed_var = v;
        }
    }

    int starting_qpos = 0;
    int best_deg = -1;
    for (size_t i = 0; i < phys_deg.size(); ++i) {
        if (phys_deg[i] > best_deg) {
            best_deg = phys_deg[i];
            starting_qpos = static_cast<int>(i);
        }
    }

    mapping_pos.assign(n_vars, -1);
    std::vector<char> assigned(n_vars, 0);
    std::vector<int> variables_assigned;
    variables_assigned.reserve(n_vars);

    std::vector<char> used_qubits(phys_adj.size(), 0);
    std::vector<int> qubits_assigned;
    qubits_assigned.reserve(n_vars);

    mapping_pos[seed_var] = starting_qpos;
    assigned[seed_var] = 1;
    variables_assigned.push_back(seed_var);
    used_qubits[starting_qpos] = 1;
    qubits_assigned.push_back(starting_qpos);

    auto neighbor_pool = [&]() {
        std::vector<int> pool;
        std::vector<char> in_pool(phys_adj.size(), 0);
        for (int qpos : qubits_assigned) {
            for (int nb : phys_adj[qpos]) {
                if (!used_qubits[nb] && !in_pool[nb]) {
                    in_pool[nb] = 1;
                    pool.push_back(nb);
                }
            }
        }
        std::sort(pool.begin(), pool.end());
        return pool;
    };

    auto distance_to_last = [&](int qpos) -> int {
        int last_pos = qubits_assigned.back();
        int last_id = qubits_ids[last_pos];
        int q_id = qubits_ids[qpos];
        int d = q_id - last_id;
        return d < 0 ? -d : d;
    };

    auto candidates = neighbor_pool();
    int remaining = n_vars - 1;
    while (remaining > 0) {
        if (candidates.empty()) {
            for (size_t i = 0; i < phys_adj.size(); ++i) {
                if (!used_qubits[i]) candidates.push_back(static_cast<int>(i));
            }
            std::sort(candidates.begin(), candidates.end());
            if (candidates.empty()) return false;
        }

        int best_col = -1;
        double best_col_score = -1e300;
        int best_col_var = -1;

        for (int qpos : candidates) {
            double col_best_score = -1e300;
            int col_best_var = -1;

            for (int v = 0; v < n_vars; ++v) {
                if (assigned[v]) continue;
                double s = 0.0;
                for (int u : variables_assigned) {
                    int mapped_pos = mapping_pos[u];
                    if (!has_neighbor(phys_adj, qpos, mapped_pos)) {
                        continue;
                    }
                    auto it = weight_adj[v].find(u);
                    if (it != weight_adj[v].end()) s += it->second;
                }
                if (s > col_best_score) {
                    col_best_score = s;
                    col_best_var = v;
                }
            }

            if (col_best_score > best_col_score) {
                best_col_score = col_best_score;
                best_col = qpos;
                best_col_var = col_best_var;
            } else if (col_best_score == best_col_score && best_col != -1) {
                if (distance_to_last(qpos) < distance_to_last(best_col)) {
                    best_col = qpos;
                    best_col_var = col_best_var;
                }
            }
        }

        int new_qpos = best_col;
        int new_v = best_col_var;
        if (new_qpos < 0 || new_v < 0) return false;

        mapping_pos[new_v] = new_qpos;
        assigned[new_v] = 1;
        variables_assigned.push_back(new_v);
        used_qubits[new_qpos] = 1;
        qubits_assigned.push_back(new_qpos);
        remaining -= 1;

        candidates = neighbor_pool();
    }

    return true;
}

static bool build_mapping_seeded_neighbor_stochastic(
    int n_vars,
    const std::vector<std::vector<int>> &phys_adj,
    const std::vector<int> &phys_deg,
    const std::vector<int> &qubits_ids,
    const std::vector<std::unordered_map<int, double>> &weight_adj,
    const std::vector<double> &abs_weight_sum,
    std::vector<int> &mapping_pos,
    std::mt19937 &rng
) {
    if (static_cast<int>(phys_adj.size()) < n_vars) return false;

    int seed_var = 0;
    double best_sum = -1.0;
    for (int v = 0; v < n_vars; ++v) {
        if (abs_weight_sum[v] > best_sum) {
            best_sum = abs_weight_sum[v];
            seed_var = v;
        }
    }

    int starting_qpos = 0;
    int best_deg = -1;
    for (size_t i = 0; i < phys_deg.size(); ++i) {
        if (phys_deg[i] > best_deg) {
            best_deg = phys_deg[i];
            starting_qpos = static_cast<int>(i);
        }
    }

    mapping_pos.assign(n_vars, -1);
    std::vector<char> assigned(n_vars, 0);
    std::vector<int> variables_assigned;
    variables_assigned.reserve(n_vars);

    std::vector<char> used_qubits(phys_adj.size(), 0);
    std::vector<int> qubits_assigned;
    qubits_assigned.reserve(n_vars);

    mapping_pos[seed_var] = starting_qpos;
    assigned[seed_var] = 1;
    variables_assigned.push_back(seed_var);
    used_qubits[starting_qpos] = 1;
    qubits_assigned.push_back(starting_qpos);

    auto neighbor_pool = [&]() {
        std::vector<int> pool;
        std::vector<char> in_pool(phys_adj.size(), 0);
        for (int qpos : qubits_assigned) {
            for (int nb : phys_adj[qpos]) {
                if (!used_qubits[nb] && !in_pool[nb]) {
                    in_pool[nb] = 1;
                    pool.push_back(nb);
                }
            }
        }
        std::sort(pool.begin(), pool.end());
        return pool;
    };

    auto candidates = neighbor_pool();
    int remaining = n_vars - 1;
    while (remaining > 0) {
        if (candidates.empty()) {
            for (size_t i = 0; i < phys_adj.size(); ++i) {
                if (!used_qubits[i]) candidates.push_back(static_cast<int>(i));
            }
            std::sort(candidates.begin(), candidates.end());
            if (candidates.empty()) return false;
        }

        std::vector<std::pair<double, int>> col_scores;
        col_scores.reserve(candidates.size());
        double best_col_score = -1e300;

        for (int qpos : candidates) {
            double best_s = -1e300;
            int best_v = -1;
            for (int v = 0; v < n_vars; ++v) {
                if (assigned[v]) continue;
                double s = 0.0;
                for (int u : variables_assigned) {
                    int mapped_pos = mapping_pos[u];
                    if (!has_neighbor(phys_adj, qpos, mapped_pos)) {
                        continue;
                    }
                    auto it = weight_adj[v].find(u);
                    if (it != weight_adj[v].end()) s += it->second;
                }
                if (s > best_s) {
                    best_s = s;
                    best_v = v;
                }
            }
            col_scores.emplace_back(best_s, best_v);
            if (best_s > best_col_score) best_col_score = best_s;
        }

        std::vector<int> top_indices;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (col_scores[i].first == best_col_score) {
                top_indices.push_back(static_cast<int>(i));
            }
        }
        if (top_indices.empty()) return false;

        std::uniform_int_distribution<int> dist(0, static_cast<int>(top_indices.size() - 1));
        int pick = top_indices[dist(rng)];
        int chosen_qpos = candidates[pick];
        int chosen_v = col_scores[pick].second;
        if (chosen_v < 0) {
            for (int v = 0; v < n_vars; ++v) {
                if (!assigned[v]) {
                    chosen_v = v;
                    break;
                }
            }
        }

        mapping_pos[chosen_v] = chosen_qpos;
        assigned[chosen_v] = 1;
        variables_assigned.push_back(chosen_v);
        used_qubits[chosen_qpos] = 1;
        qubits_assigned.push_back(chosen_qpos);
        remaining -= 1;

        candidates = neighbor_pool();
    }

    return true;
}

static bool build_mapping_semantic_group_then_fill(
    int n_vars,
    const std::vector<int> &priority_vars,
    const std::vector<std::vector<int>> &logical_adj,
    const std::vector<int> &log_deg,
    const std::vector<std::vector<int>> &phys_adj,
    const std::vector<int> &phys_deg,
    std::vector<int> &mapping_pos,
    std::mt19937 &rng,
    const std::string &remainder
) {
    if (static_cast<int>(phys_adj.size()) < n_vars) return false;

    std::vector<char> seen(n_vars, 0);
    std::vector<int> prio;
    prio.reserve(priority_vars.size());
    for (int v : priority_vars) {
        if (v < 0 || v >= n_vars) continue;
        if (!seen[v]) {
            prio.push_back(v);
            seen[v] = 1;
        }
    }

    std::vector<int> remaining_vars;
    remaining_vars.reserve(n_vars - prio.size());
    for (int v = 0; v < n_vars; ++v) {
        if (!seen[v]) remaining_vars.push_back(v);
    }

    std::vector<int> qubits_order = order_by_degree_desc(phys_deg);
    std::vector<int> top_qubits;
    top_qubits.reserve(prio.size());
    for (size_t i = 0; i < prio.size(); ++i) {
        top_qubits.push_back(qubits_order[i]);
    }

    std::vector<int> prio_order = prio;
    std::sort(prio_order.begin(), prio_order.end(), [&log_deg](int a, int b) {
        if (log_deg[a] != log_deg[b]) return log_deg[a] > log_deg[b];
        return a < b;
    });

    mapping_pos.assign(n_vars, -1);
    std::vector<char> used(phys_adj.size(), 0);

    auto score_place = [&](int v, int qpos) -> int {
        int s = 0;
        for (int u : logical_adj[v]) {
            int mapped = mapping_pos[u];
            if (mapped < 0) continue;
            if (has_neighbor(phys_adj, qpos, mapped)) {
                s += 1;
            }
        }
        return s;
    };

    for (int v : prio_order) {
        int best_qpos = -1;
        int best_s = -1;
        for (int qpos : top_qubits) {
            if (used[qpos]) continue;
            int s = score_place(v, qpos);
            if (s > best_s) {
                best_s = s;
                best_qpos = qpos;
            }
        }
        if (best_qpos < 0) {
            for (int qpos : top_qubits) {
                if (!used[qpos]) {
                    best_qpos = qpos;
                    break;
                }
            }
        }
        if (best_qpos < 0) return false;
        mapping_pos[v] = best_qpos;
        used[best_qpos] = 1;
    }

    if (remaining_vars.empty()) return true;

    if (remainder == "random") {
        std::vector<int> remaining_qubits;
        remaining_qubits.reserve(phys_adj.size());
        for (size_t i = 0; i < phys_adj.size(); ++i) {
            if (!used[i]) remaining_qubits.push_back(static_cast<int>(i));
        }
        std::shuffle(remaining_qubits.begin(), remaining_qubits.end(), rng);
        if (remaining_qubits.size() < remaining_vars.size()) return false;
        for (size_t i = 0; i < remaining_vars.size(); ++i) {
            mapping_pos[remaining_vars[i]] = remaining_qubits[i];
        }
        return true;
    }

    if (remainder != "greedy") return false;

    int total_vars = static_cast<int>(prio.size() + remaining_vars.size());
    int K = std::min(static_cast<int>(phys_adj.size()), std::max(128, 8 * std::max(1, total_vars)));

    std::vector<int> cand_qubits;
    for (int i = 0; i < K; ++i) {
        int qpos = qubits_order[i];
        if (!used[qpos]) cand_qubits.push_back(qpos);
    }
    if (cand_qubits.size() < remaining_vars.size()) {
        cand_qubits.clear();
        for (size_t i = 0; i < phys_adj.size(); ++i) {
            if (!used[i]) cand_qubits.push_back(static_cast<int>(i));
        }
    }

    std::vector<int> rem_order = remaining_vars;
    std::sort(rem_order.begin(), rem_order.end(), [&log_deg](int a, int b) {
        if (log_deg[a] != log_deg[b]) return log_deg[a] > log_deg[b];
        return a < b;
    });

    for (int v : rem_order) {
        int best_qpos = -1;
        int best_s = -1;
        for (int qpos : cand_qubits) {
            if (used[qpos]) continue;
            int s = score_place(v, qpos);
            if (s > best_s) {
                best_s = s;
                best_qpos = qpos;
            }
        }
        if (best_qpos < 0) {
            for (size_t i = 0; i < phys_adj.size(); ++i) {
                if (!used[i]) {
                    best_qpos = static_cast<int>(i);
                    break;
                }
            }
        }
        if (best_qpos < 0) return false;
        mapping_pos[v] = best_qpos;
        used[best_qpos] = 1;
    }

    return true;
}

} // namespace

extern "C" const char *embed_last_error() {
    return g_last_error.c_str();
}

extern "C" int embed_no_chains_drop_missing_cpp(
    int n_vars,
    const int *quad_u,
    const int *quad_v,
    const double *quad_w,
    int m,
    const double *linear_bias,
    const int *nodelist,
    int n_nodes,
    const int *edge_u,
    const int *edge_v,
    int n_edges,
    const char *strategy_c,
    const int *priority_vars,
    int n_priority,
    unsigned int rng_seed,
    int *out_mapping,
    int *out_linear_qubit,
    double *out_linear_bias,
    int *out_keep_u,
    int *out_keep_v,
    double *out_keep_w,
    int *out_kept_count,
    double *out_mapping_ms,
    double *out_embed_ms
) {
    try {
        if (n_vars <= 0 || n_nodes <= 0) {
            g_last_error = "invalid sizes";
            return 1;
        }
        if (!quad_u || !quad_v || !quad_w || !linear_bias || !nodelist || !edge_u || !edge_v || !strategy_c) {
            g_last_error = "null input";
            return 2;
        }
        if (!out_mapping || !out_linear_qubit || !out_linear_bias || !out_keep_u || !out_keep_v || !out_keep_w) {
            g_last_error = "null output";
            return 3;
        }

        std::string strategy = strategy_c ? std::string(strategy_c) : std::string();

        std::vector<int> qubit_ids(n_nodes);
        qubit_ids.reserve(n_nodes);
        std::unordered_map<int, int> id_to_pos;
        id_to_pos.reserve(static_cast<size_t>(n_nodes) * 2);
        for (int i = 0; i < n_nodes; ++i) {
            int qid = nodelist[i];
            qubit_ids[i] = qid;
            id_to_pos[qid] = i;
        }

        std::vector<std::vector<int>> phys_adj(n_nodes);
        std::vector<int> phys_deg(n_nodes, 0);
        std::unordered_set<int64_t> phys_edge_set;
        phys_edge_set.reserve(static_cast<size_t>(n_edges) * 2 + 16);

        for (int i = 0; i < n_edges; ++i) {
            int a_id = edge_u[i];
            int b_id = edge_v[i];
            auto it_a = id_to_pos.find(a_id);
            auto it_b = id_to_pos.find(b_id);
            if (it_a == id_to_pos.end() || it_b == id_to_pos.end()) continue;
            int a_pos = it_a->second;
            int b_pos = it_b->second;
            phys_adj[a_pos].push_back(b_pos);
            phys_adj[b_pos].push_back(a_pos);
            phys_deg[a_pos] += 1;
            phys_deg[b_pos] += 1;
            phys_edge_set.insert(edge_key(a_id, b_id));
        }
        for (auto &neighbors : phys_adj) {
            std::sort(neighbors.begin(), neighbors.end());
        }

        std::vector<std::vector<int>> logical_adj;
        std::vector<int> log_deg;
        build_logical_adj(n_vars, quad_u, quad_v, m, logical_adj, log_deg);

        std::vector<std::unordered_map<int, double>> weight_adj;
        std::vector<double> abs_weight_sum;
        build_weight_adj(n_vars, quad_u, quad_v, quad_w, m, weight_adj, abs_weight_sum);

        std::vector<int> mapping_pos;
        std::vector<int> qubits_order_pos = order_by_degree_desc(phys_deg);

        std::mt19937 rng(rng_seed == 0 ? static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()) : rng_seed);

        auto t_map_start = std::chrono::steady_clock::now();
        bool ok = false;

        SemanticParse parsed = parse_semantic(strategy);
        if (parsed.ok) {
            if (!priority_vars && n_priority > 0) {
                g_last_error = "semantic strategy priority vars pointer is null";
                return 4;
            }
            std::vector<int> prio;
            prio.reserve(n_priority);
            for (int i = 0; i < n_priority; ++i) prio.push_back(priority_vars[i]);
            ok = build_mapping_semantic_group_then_fill(
                n_vars,
                prio,
                logical_adj,
                log_deg,
                phys_adj,
                phys_deg,
                mapping_pos,
                rng,
                parsed.remainder
            );
        } else if (strategy == "random") {
            ok = build_mapping_random(n_vars, qubits_order_pos, mapping_pos, rng);
        } else if (strategy == "greedy") {
            ok = build_mapping_greedy(n_vars, logical_adj, log_deg, phys_adj, qubits_order_pos, mapping_pos);
        } else if (strategy == "seeded_neighbor_greedy") {
            ok = build_mapping_seeded_neighbor_greedy(n_vars, phys_adj, phys_deg, qubit_ids, weight_adj, abs_weight_sum, mapping_pos);
        } else if (strategy == "seeded_neighbor_stochastic") {
            ok = build_mapping_seeded_neighbor_stochastic(n_vars, phys_adj, phys_deg, qubit_ids, weight_adj, abs_weight_sum, mapping_pos, rng);
        } else {
            g_last_error = "unknown strategy";
            return 5;
        }

        auto t_map_end = std::chrono::steady_clock::now();
        if (!ok) {
            g_last_error = "mapping failed";
            return 6;
        }

        for (int v = 0; v < n_vars; ++v) {
            int qpos = mapping_pos[v];
            if (qpos < 0 || qpos >= n_nodes) {
                g_last_error = "invalid mapping";
                return 7;
            }
            int qid = qubit_ids[qpos];
            out_mapping[v] = qid;
            out_linear_qubit[v] = qid;
            out_linear_bias[v] = linear_bias[v];
        }

        auto t_embed_start = std::chrono::steady_clock::now();
        int kept = 0;
        for (int i = 0; i < m; ++i) {
            int u = quad_u[i];
            int v = quad_v[i];
            if (u < 0 || v < 0 || u >= n_vars || v >= n_vars || u == v) {
                continue;
            }
            int mu = out_mapping[u];
            int mv = out_mapping[v];
            if (phys_edge_set.find(edge_key(mu, mv)) == phys_edge_set.end()) {
                continue;
            }
            out_keep_u[kept] = mu;
            out_keep_v[kept] = mv;
            out_keep_w[kept] = quad_w[i];
            kept += 1;
        }

        auto t_embed_end = std::chrono::steady_clock::now();

        if (out_kept_count) *out_kept_count = kept;
        if (out_mapping_ms) {
            *out_mapping_ms = std::chrono::duration<double, std::milli>(t_map_end - t_map_start).count();
        }
        if (out_embed_ms) {
            *out_embed_ms = std::chrono::duration<double, std::milli>(t_embed_end - t_embed_start).count();
        }

        return 0;
    } catch (const std::exception &e) {
        g_last_error = e.what();
        return 100;
    } catch (...) {
        g_last_error = "unknown error";
        return 101;
    }
}
