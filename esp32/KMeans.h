#ifndef KMEANS_H
#define KMEANS_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

class KMeans {
  private:
    int k;                      // Number of clusters
    int max_iters;              // Maximum number of iterations
    float tol;                  // Convergence tolerance
    std::vector<std::vector<float>> centroids; // Learned cluster centers (k x dims)
    std::mt19937 rng;

    // Euclidean distance squared between two points
    float euclideanDistanceSq(const std::vector<float>& a, const std::vector<float>& b) const {
      float sum = 0.0f;
      for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
      }
      return sum;
    }

    // K-Means++ initialization for better initial centroids
    void initKMeansPlusPlus(const std::vector<std::vector<float>>& dataset) {
      int n_total = dataset.size();
      int dims = dataset[0].size();
      centroids.clear();
      centroids.reserve(k);

      // 1. Choose first centroid uniformly at random
      std::uniform_int_distribution<int> dist_idx(0, n_total - 1);
      centroids.push_back(dataset[dist_idx(rng)]);

      // 2. Choose remaining k-1 centroids
      std::vector<float> minDistSq(n_total, std::numeric_limits<float>::max());

      for (int i = 1; i < k; ++i) {
        float sumDistSq = 0.0f;
        for (int j = 0; j < n_total; ++j) {
          float dsq = euclideanDistanceSq(dataset[j], centroids.back());
          if (dsq < minDistSq[j]) {
            minDistSq[j] = dsq;
          }
          sumDistSq += minDistSq[j];
        }

        // Pick next centroid with probability proportional to D(x)^2
        std::uniform_real_distribution<float> dist_prob(0.0f, sumDistSq);
        float target = dist_prob(rng);
        float currentSum = 0.0f;
        int chosen_idx = 0;

        for (int j = 0; j < n_total; ++j) {
          currentSum += minDistSq[j];
          if (currentSum >= target) {
            chosen_idx = j;
            break;
          }
        }
        centroids.push_back(dataset[chosen_idx]);
      }
    }

  public:
    KMeans(int num_clusters = 3, int iterations = 100, float tolerance = 1e-4f)
      : k(num_clusters), max_iters(iterations), tol(tolerance) {

#ifdef ESP32
      rng.seed(esp_random());
#else
      rng.seed(std::random_device()());
#endif
    }

    ~KMeans() = default;

    // -- TRAINING -- //
    void train(const std::vector<std::vector<float>>& dataset) {
      if (dataset.empty() || k <= 0) return;

      int n_total = dataset.size();
      int dims = dataset[0].size();

      // Adjust k if dataset is smaller than k
      if (n_total < k) {
        k = n_total;
      }

      // Initialize centroids using K-Means++
      initKMeansPlusPlus(dataset);

      std::vector<int> assignments(n_total);
      std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dims, 0.0f));
      std::vector<int> counts(k, 0);

      for (int iter = 0; iter < max_iters; ++iter) {
        // 1. Expectation Step: Assign each point to the nearest centroid
        for (int i = 0; i < n_total; ++i) {
          float min_dist = std::numeric_limits<float>::max();
          int best_cluster = 0;
          for (int j = 0; j < k; ++j) {
            float dist = euclideanDistanceSq(dataset[i], centroids[j]);
            if (dist < min_dist) {
              min_dist = dist;
              best_cluster = j;
            }
          }
          assignments[i] = best_cluster;
        }

        // 2. Maximization Step: Recalculate centroids
        for (int j = 0; j < k; ++j) {
          std::fill(new_centroids[j].begin(), new_centroids[j].end(), 0.0f);
          counts[j] = 0;
        }

        for (int i = 0; i < n_total; ++i) {
          int c_idx = assignments[i];
          counts[c_idx]++;
          for (int d = 0; d < dims; ++d) {
            new_centroids[c_idx][d] += dataset[i][d];
          }
        }

        float max_shift = 0.0f;
        for (int j = 0; j < k; ++j) {
          if (counts[j] > 0) {
            for (int d = 0; d < dims; ++d) {
              new_centroids[j][d] /= counts[j];
            }
            // Measure shift for convergence check
            max_shift = std::max(max_shift, std::sqrt(euclideanDistanceSq(centroids[j], new_centroids[j])));
            centroids[j] = new_centroids[j];
          }
        }

        // Check convergence
        if (max_shift < tol) {
          break;
        }
      }
    }

    // -- INFERENCE -- //
    // Returns the cluster index (0 to k-1) for a given sample.
    int predict(const std::vector<float>& sample) const {
      if (centroids.empty()) return -1;

      float min_dist = std::numeric_limits<float>::max();
      int best_cluster = 0;

      for (size_t j = 0; j < centroids.size(); ++j) {
        float dist = euclideanDistanceSq(sample, centroids[j]);
        if (dist < min_dist) {
          min_dist = dist;
          best_cluster = j;
        }
      }
      return best_cluster;
    }

    // Returns the Euclidean distance from the sample to its assigned centroid.
    float distanceToCentroid(const std::vector<float>& sample) const {
      int cluster = predict(sample);
      if (cluster < 0) return -1.0f;
      return std::sqrt(euclideanDistanceSq(sample, centroids[cluster]));
    }

    // Get reference to the learned centroids
    const std::vector<std::vector<float>>& getCentroids() const {
      return centroids;
    }

    // Save trained model to a binary file
    bool save(const std::string& filename) const {
      std::ofstream out(filename, std::ios::binary);
      if (!out.is_open()) return false;

      // Save metadata
      int dims = centroids.empty() ? 0 : centroids[0].size();
      out.write(reinterpret_cast<const char*>(&k), sizeof(k));
      out.write(reinterpret_cast<const char*>(&max_iters), sizeof(max_iters));
      out.write(reinterpret_cast<const char*>(&dims), sizeof(dims));

      // Save centroids data
      for (int i = 0; i < k; ++i) {
        out.write(reinterpret_cast<const char*>(centroids[i].data()), dims * sizeof(float));
      }
      return true;
    }

    // Load trained model from a binary file
    bool load(const std::string& filename) {
      std::ifstream in(filename, std::ios::binary);
      if (!in.is_open()) return false;

      int dims = 0;
      in.read(reinterpret_cast<char*>(&k), sizeof(k));
      in.read(reinterpret_cast<char*>(&max_iters), sizeof(max_iters));
      in.read(reinterpret_cast<char*>(&dims), sizeof(dims));

      centroids.assign(k, std::vector<float>(dims, 0.0f));
      for (int i = 0; i < k; ++i) {
        in.read(reinterpret_cast<char*>(centroids[i].data()), dims * sizeof(float));
      }
      return true;
    }
};

#endif

