#include "octomap_compare/random_forest_classifier.h"

#include <chrono>
#include <fstream>
#include <numeric>

RandomForestClassifier::RandomForestClassifier(const Params &params) :
    params_(params), feature_extractor_(params.histogram_params) {
  load(params_.classifier_file_name);
}

void RandomForestClassifier::classify(const std::vector<Cluster>& clusters,
                                      std::vector<bool>* labels) {
  const size_t n_clusters = clusters.size();
  CHECK_NOTNULL(labels)->assign(n_clusters, false);

  cv::Mat features(1, FeatureExtractor::kNFeatures, CV_32FC1);;

  for (size_t i = 0; i < n_clusters; ++i) {
    // Get features.
    feature_extractor_.getClusterFeatures(clusters[i], &features);
    // Classify.
    const double prob = rtrees_.predict_prob(features);
    if (prob > params_.probability_threshold) {
      labels->at(i) = true;
      LOG(INFO) << "Cluster with id " << clusters[i].id << " classified dynamic.";
    }
  }
}

void RandomForestClassifier::load(const std::string &filename) {
  LOG(INFO)<< "Loading classifier from: " << filename << ".";
  rtrees_.load(filename.c_str());
}

void RandomForestClassifier::save(const std::string &filename) const {
  LOG(INFO)<< "Saving classifier to: " << filename << ".";
  rtrees_.save(filename.c_str());
}

void RandomForestClassifier::train(const std::vector<Cluster>& clusters,
                                   const std::vector<bool> &labels) {
  CHECK_EQ(clusters.size(), labels.size()) << "Training data and labels have different size.";
  CHECK_NE(labels.size(), 0) << "Got empty training labels.";
  CHECK_GT(sumOfEqualValues(labels, true), 0) << "Training data has no positive labels.";
  CHECK_GT(sumOfEqualValues(labels, false), 0) << "Training data has no negative labels.";

  const size_t n_clusters = clusters.size();
  cv::Mat features(n_clusters, FeatureExtractor::kNFeatures, CV_32FC1);
  LOG(INFO) << "Input is " << n_clusters << " training samples.";

  // Get features.
  feature_extractor_.getClusterFeatures(clusters, &features);

  // Write labels to OpenCV matrix.
  cv::Mat cv_labels(n_clusters, 1, CV_32SC1);
  for (size_t i = 0; i < n_clusters; ++i) {
    cv_labels.at<int>(i,0) = (int)labels[i];
  }

  // Create parameter object.
  const float priors[] = { params_.rf_priors[0], params_.rf_priors[1] };
  CvRTParams rtrees_params = CvRTParams(
      params_.rf_max_depth, params_.rf_min_sample_ratio * n_clusters,
      params_.rf_regression_accuracy, params_.rf_use_surrogates,
      params_.rf_max_categories, priors, params_.rf_calc_var_importance,
      params_.rf_n_active_vars, params_.rf_max_num_of_trees,
      params_.rf_accuracy,
      CV_TERMCRIT_ITER+CV_TERMCRIT_EPS);
  LOG(INFO) << "Training....";
  std::cout << "Training (Choo-Choo)... " << std::flush;  // Print to console to show that something's happening.
  rtrees_.train(features, CV_ROW_SAMPLE, cv_labels, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
                rtrees_params);
//  rtrees_.train(features, CV_ROW_SAMPLE, cv_labels);

  if (params_.save_classifier_after_training) save(params_.classifier_file_name);

}

std::vector<bool> probGreaterThan(const std::vector<double>& probabilities,
                                  const double& prob_threshold) {
  const size_t vec_size = probabilities.size();
  std::vector<bool> result(vec_size);
  for (size_t i = 0; i < vec_size; ++i) {
    result[i] = probabilities[i] > prob_threshold;
  }
  return result;
}

inline double getMaxClusterDist(const Cluster& cluster) {
  double max_dist = 0;
  for (const ClusterPoint& point: cluster.points) {
    if (point.distance > max_dist) max_dist = point.distance;
  }
  return max_dist;
}

inline double getMinClusterDist(const Cluster& cluster) {
  double min_dist = std::numeric_limits<double>::max();
  for (const ClusterPoint& point: cluster.points) {
    if (point.distance < min_dist) min_dist = point.distance;
  }
  return min_dist;
}

inline double getMeanClusterDist(const Cluster& cluster) {
  double mean = 0;
  for (const ClusterPoint& point: cluster.points) {
    mean += point.distance;
  }
  mean /= cluster.points.size();
  return mean;
}

void printCompROC(const std::vector<Cluster>&clusters, const std::vector<bool>& labels) {
  std::ofstream file;
  file.open("/tmp/comp_heat_vals.csv");
  CHECK(file.is_open()) << "Could not open file.";
  size_t i = 0;
  for (const Cluster& cluster: clusters) {
    file << getMinClusterDist(cluster) << ", "
         << getMeanClusterDist(cluster) << ", "
         << getMaxClusterDist(cluster) << ", "
         << labels[i++] << "\n";
  }
  file.close();
}

void RandomForestClassifier::test(const std::vector<Cluster>& clusters,
                                  const std::vector<bool>& labels) {
  const size_t n_clusters = clusters.size();
  CHECK_EQ(clusters.size(), labels.size()) << "Clusters and labels have different size.";
  CHECK_GT(sumOfEqualValues(labels, true), 0) << "Test data has no positive labels.";
  CHECK_GT(sumOfEqualValues(labels, false), 0) << "Test data has no negative labels.";
  printCompROC(clusters, labels);

  cv::Mat features(1, FeatureExtractor::kNFeatures, CV_32FC1);
  std::vector<double> pred_prob(n_clusters);

  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < n_clusters; ++i) {
    // Get features.
    feature_extractor_.getClusterFeatures(clusters[i], &features);
    // Predict.
    pred_prob[i] = rtrees_.predict_prob(features);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // Get ROC curve values.
  static constexpr unsigned int kNROCSteps = 100u;
  static constexpr float kROCResolution = 1.0 / kNROCSteps;
  std::vector<std::pair<float, float> > roc_vec(kNROCSteps);
  roc_vec.front() = std::make_pair(1.0, 1.0);
  roc_vec.back() = std::make_pair(0.0, 0.0);
  float area_under_curve = 0;
  float best_threshold = 0;
  float max_accuracy = 0;
  for (size_t i = 1; i < kNROCSteps - 1u; ++i) {
    // Get false pos and true pos.
    const std::vector<bool> pred_labels = probGreaterThan(pred_prob, kROCResolution*i);
    roc_vec[i] = getFalsePosAndTruePos(pred_labels, labels);

    // Update best threshold.
    const float accuracy = -roc_vec[i].first + roc_vec[i].second;
    if (accuracy > max_accuracy) {
      max_accuracy = accuracy;
      best_threshold = kROCResolution*i;
    }

    // Compute area under curve increment.
    const float delta_x = roc_vec[i-1].first - roc_vec[i].first;
    const float y_val = (roc_vec[i-1].second + roc_vec[i].second) / 2;
    CHECK(delta_x >= 0) << "Something went wrong computing area under curve increment.";
    if (delta_x > 0) {
      area_under_curve += y_val * delta_x;
    }
    LOG(INFO) << roc_vec[i].first << " " << roc_vec[i].second;
  }
  LOG(INFO) << "Area under curve is " << area_under_curve << "\n";
  LOG(INFO) << "Prediction took " << duration.count() * 1000 << "ms total -- "
                                  << duration.count() * 1000 / n_clusters << "ms per prediction.";

  // Print current precision/recall.
  const std::vector<bool> cur_param_labels =
      probGreaterThan(pred_prob, params_.probability_threshold);
  std::pair<float, float> prec_rec = getPrecisionAndRecall(cur_param_labels, labels);
  LOG(INFO) << "Precision/Recall for current threshold " << params_.probability_threshold
            << " is " << prec_rec.first << "/" << prec_rec.second << "\n";

  // Print best precision/recall.
  const std::vector<bool> best_param_labels =
      probGreaterThan(pred_prob, best_threshold);
  std::pair<float, float> best_prec_rec = getPrecisionAndRecall(best_param_labels, labels);
  LOG(INFO) << "Precision/Recall for best threshold " << best_threshold
            << " is " << best_prec_rec.first << "/" << best_prec_rec.second << "\n";

  // Write ROC curve to file.
  std::ofstream file;
  file.open(params_.roc_filename);
  if (file.is_open()) {
    for (const auto& prec_rec: roc_vec) {
      file << prec_rec.first << ", " << prec_rec.second << "\n";
    }
    file.close();
  }
  else {
    LOG(ERROR) << "Could not open file " << params_.roc_filename << ".";
  }
}
