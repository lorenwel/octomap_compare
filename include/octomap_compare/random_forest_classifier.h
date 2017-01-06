#ifndef OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_
#define OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_

#include "octomap_compare/octomap_compare_utils.h"

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "octomap_compare/octomap_compare_utils.h"

class FeatureExtractor {

public:
  struct HistogramParams {
    double min_val;
    size_t n_bins;
    double bin_size;
  };

private:

  const HistogramParams params_;

  void getEigenvalueFeatures(Eigen::Vector3d eig,
                             std::vector<double>* eigenvalue_features);

  // Counts number of occurences for every histogram bin. All values greater than the maximum
  // that fits into the histogram are pooled into the last bin.
  void getHistogram(const Cluster& cluster, std::vector<size_t>* histogram);

  // Returns density of points in <sphere around mean, circle around mean>.
  std::pair<double, double> getDensities(Eigen::MatrixXd cartesian,
                                         const Eigen::Vector3d& normal);

public:

  FeatureExtractor(const HistogramParams& params) : params_(params) {}

  void getClusterFeatures(const Cluster& cluster, cv::Mat* features);

};

#endif /* OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_ */
