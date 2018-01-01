#include <random>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>

static constexpr double pi = 3.14159265358979323846;
static std::uniform_real_distribution<double> rand01(0.0, 1.0);

inline double normal_density(double mean, double stddev, double x) {
    auto inv_stddev = 1.0 / stddev;
    auto k = (x - mean) * inv_stddev;
    return (1.0 / std::sqrt(2 * pi)) * inv_stddev * std::exp(-0.5 * k * k);
}

template <class Generator>
uint32_t choose(uint32_t min, uint32_t max, Generator& gen) {
    std::uniform_int_distribution<uint32_t> distrib(min, max);
    return distrib(gen);
}

// Mixture of gaussians
class Mixture {
public:
    Mixture()
        : weights_({1.0})
        , gaussians_({{0.0, 1.0}})
    {
        normalize();
    }

    Mixture(std::vector<double>&& weights,
            std::vector<std::pair<double, double>>&& gaussians)
        : weights_(std::move(weights))
        , gaussians_(std::move(gaussians))
    {
        normalize();
    }

    template <class Generator>
    double simulate(Generator& gen) {
        auto index = select_(gen);
        return distribs_[index](gen);
    }

    double evaluate(double x) const {
        double sum = 0.0;
        for (size_t i = 0, n = size(); i < n; i++)
            sum += weights_[i] * normal_density(gaussians_[i].first, gaussians_[i].second, x); 
        return sum;
    }

    void dump() const {
        for (size_t i = 0; i < weights_.size(); i++)
            std::cout << weights_[i] << ": " << gaussians_[i].first << " " << gaussians_[i].second << "\n";
    }

    size_t size() const { return distribs_.size(); }
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<std::pair<double, double>>& gaussians() const { return gaussians_; }

private:
    void normalize() {
        const size_t n = weights_.size();
        assert(n == gaussians_.size());

        // Normalize the weights
        auto total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        auto inv_total = 1.0 / total;
        for (auto& w : weights_) w *= inv_total;

        // Order the gaussians by their mean
        std::vector<uint32_t> ids(n);
        for (size_t i = 0; i < n; i++) ids[i] = i;
        std::sort(ids.begin(), ids.end(), [&] (uint32_t a, uint32_t b) {
            return gaussians_[a].first < gaussians_[b].first;
        });
        std::vector<double> new_weights(n);
        std::vector<std::pair<double, double>> new_gaussians(n);
        for (size_t i = 0; i < n; i++) {
            new_weights[i]   = weights_[ids[i]];
            new_gaussians[i] = gaussians_[ids[i]];
        }
        std::swap(weights_, new_weights);
        std::swap(gaussians_, new_gaussians);
    
        // Create the normal distributions and the selection distribution    
        for (auto g : gaussians_)
            distribs_.emplace_back(g.first, g.second);
        select_ = std::discrete_distribution<uint32_t>(weights_.begin(), weights_.end());
    }
    
    std::vector<double> weights_;
    std::vector<std::pair<double, double>> gaussians_;

    std::discrete_distribution<uint32_t> select_;
    std::vector<std::normal_distribution<double>> distribs_;
};

// Holds the data necessary to perform an
// iteration of the reversible jump algorithm
struct RevJump {
    std::vector<double>& data;
    double old_l;
    Mixture mixture;

    // Intra-model jump
    std::normal_distribution<double> weight_jump;
    std::normal_distribution<double> mean_jump;
    std::normal_distribution<double> stddev_jump;
 
    RevJump(std::vector<double>& data, Mixture&& mixture = Mixture())
        : data(data)
        , mixture(std::move(mixture))
        , weight_jump(0.0, 0.2)
        , mean_jump(0.0, 0.2)
        , stddev_jump(0.0, 0.4)
    {
        old_l = log_likelihood(this->mixture);
    }

    double log_likelihood(const Mixture& mixture) {
        double sum = 0.0;
        for (auto d : data)
            sum += std::log(mixture.evaluate(d));
        return sum;
    }

    template <class Generator>
    void jump(Generator& gen) {
        if (choose(0, 9, gen) != 9) {
            // Intra-model jump
            intra_jump(gen);
        } else {
            // Inter-model jump
            inter_jump(gen);
        }
    }

    template <class Generator>
    void intra_jump(Generator& gen) {
        // 3 cases:
        // (1) updating the weights
        // (2) updating the mean
        // (3) updating the deviation
        auto weights   = mixture.weights();
        auto gaussians = mixture.gaussians();

        // Symmetric proposals (i.e. q(x|y) = q(y|x))
        uint32_t index  = choose(0, mixture.size() - 1, gen);
        uint32_t update = choose(0, 2, gen);
        switch (update) {
            case 0: weights[index]          += weight_jump(gen); break;
            case 1: gaussians[index].first  += mean_jump(gen);   break;
            case 2: gaussians[index].second += stddev_jump(gen); break;
            default: assert(false); break;
        }
        weights[index] = std::fabs(weights[index]);
        gaussians[index].second = std::fabs(gaussians[index].second);

        Mixture new_mixture(std::move(weights), std::move(gaussians));
        auto new_l = log_likelihood(new_mixture);
        auto accept = std::min(1.0, std::exp(new_l - old_l));
        if (rand01(gen) < accept) {
            mixture = new_mixture;
            old_l   = new_l;
        }
    }

    template <class Generator>
    void inter_jump(Generator& gen) {
        const double model_prior = 1.0e5;
        if (choose(0, 1, gen) == 0) {
            // split
            auto index = choose(0, mixture.size() - 1, gen);
            auto split = mixture.size();
            auto u1 = rand01(gen);
            auto u2 = rand01(gen);
            auto u3 = rand01(gen);
            if (u1 == 1.0) u1 = 0.0;
            
            auto weights   = mixture.weights();
            auto gaussians = mixture.gaussians();
            weights.emplace_back(weights[index]);
            gaussians.emplace_back(gaussians[index]);

            auto w = weights[index];
            auto m = gaussians[index].first;
            auto d = gaussians[index].second;
            weights[index] *= u1;
            gaussians[index].first *= u2;
            gaussians[index].second *= std::sqrt(u3);
            weights[split] *= 1.0 - u1;
            gaussians[split].first *= (1.0 - u1 * u2) / (1.0 - u1);
            gaussians[split].second *= std::sqrt((1.0 - u1 * u3) / (1.0 - u1));

            Mixture new_mixture(std::move(weights), std::move(gaussians));
            auto new_l = log_likelihood(new_mixture);
            auto accept = std::min(1.0,
                std::exp(new_l - old_l) *
                (1.0 / model_prior) * // Prior on mixture of dim k = (1.0 / model_prior)^k
                // Jacobian (see Robert & Casella "Monte Carlo Statistical Methods" p.439)
                w * w * w *
                std::fabs(m) *
                d * d / ((1.0 - u1) * (1.0 - u1))
            );
            if (rand01(gen) < accept) {
                mixture = new_mixture;
                old_l = new_l;
            }
        } else {
            // merge
            if (mixture.size() < 2) return;

            auto index = choose(0, mixture.size() - 1, gen);
            auto split = index;
            while (split == index) split = choose(0, mixture.size() - 1, gen);

            auto weights   = mixture.weights();
            auto gaussians = mixture.gaussians();
            auto invert_transform = [&] (double x, double y) {
                return ((weights[index] - weights[split]) * y + x * weights[split]) * (1.0 / weights[index]);
            };
            auto w = weights[index] + weights[split];
            auto m = invert_transform(gaussians[index].first, gaussians[split].first);
            auto d = std::sqrt(std::fabs(invert_transform(
                gaussians[index].second * gaussians[index].second,
                gaussians[split].second * gaussians[split].second
                ))
            );
            auto u1 = weights[index] / w;
            weights[index] = w;
            gaussians[index].first = m;
            gaussians[index].second = d;
            std::swap(weights[split], weights.back());
            std::swap(gaussians[split], gaussians.back());
            weights.pop_back();
            gaussians.pop_back();

            Mixture new_mixture(std::move(weights), std::move(gaussians));
            auto new_l = log_likelihood(new_mixture);
            auto accept = std::min(1.0,
                std::exp(new_l - old_l) *
                model_prior * // Prior on mixture of dim k = (1.0 / model_prior)^k
                // Inverse Jacobian
                (1.0 - u1) * (1.0 - u1) / (w * w * w * std::fabs(m) * d * d)
            );
            if (rand01(gen) < accept) {
                mixture = new_mixture;
                old_l = new_l;
            }
        }
    }
};

int main(int argc, char** argv) {
    const size_t N = 40000;
    const size_t M = 5000;
    const uint32_t seed = 123456;

    // Simulate some data from the target mixture
    Mixture mixture({0.1, 0.1, 0.4, 0.4},
                    {{0.1, 0.02}, {0.4, 0.01}, {0.8, 0.05}, {2.0, 0.01}});
    std::mt19937 gen(seed);
    std::vector<double> data(N);
    for (size_t i = 0; i < N; i++) data[i] = mixture.simulate(gen);

    // Setup variables for the reversible jump algo.
    Mixture init({1.0},
                 {{0.0, 0.5}});
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<std::pair<double, double>>> gaussians;
    std::vector<size_t> count;

    // Run M iterations
    RevJump rev_jump(data, std::move(init));
    for (size_t i = 0; i < M; i++) {
        rev_jump.jump(gen);

        auto d = rev_jump.mixture.size();
        if (weights.size() <= d) {
            weights.resize(d + 1);
            gaussians.resize(d + 1);
            count.resize(d + 1, 0); 
            weights[d].resize(d, 0.0);
            gaussians[d].resize(d, std::make_pair(0.0, 0.0));
        }
        for (size_t i = 0; i < d; i++) {
            weights[d][i]          += rev_jump.mixture.weights()[i];
            gaussians[d][i].first  += rev_jump.mixture.gaussians()[i].first;
            gaussians[d][i].second += rev_jump.mixture.gaussians()[i].second;
        }
        count[d]++;
    }

    // Normalize the results
    for (size_t d = 0; d < weights.size(); d++) {
        if (count[d] == 0) continue;
        for (size_t i = 0; i < d; i++) {
            weights[d][i]          /= count[d];
            gaussians[d][i].first  /= count[d];
            gaussians[d][i].second /= count[d];
        }
    } 

    std::cout << "Target mixture:\n" 
              << "L(theta|x) = " << rev_jump.log_likelihood(mixture) << std::endl;
    mixture.dump();

    for (size_t d = 1; d < weights.size(); d++) {
        if (count[d] == 0) continue;
        std::cout << d << ": " << count[d] << " ----------------------------------------\n";
        Mixture estimate(std::move(weights[d]), std::move(gaussians[d]));
        estimate.dump();
        std::cout << "L(theta|x) = " << rev_jump.log_likelihood(estimate) << std::endl;
    }
    //for (auto d : data) std::cout << d << '\n';
    return 0;
}
