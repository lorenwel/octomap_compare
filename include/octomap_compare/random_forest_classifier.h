#ifndef OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_
#define OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "octomap_compare/feature_extractor.h"
#include "octomap_compare/octomap_compare_utils.h"

class RandomForestClassifier {
public:
  struct Params {
    FeatureExtractor::HistogramParams histogram_params;
    double probability_threshold;
    bool save_classifier_after_training;
    std::string classifier_file_name;
    std::string roc_filename;

    // OpenCv random forest parameters.
    int rf_max_depth;
    double rf_min_sample_ratio;
    double rf_regression_accuracy;
    bool rf_use_surrogates;
    int rf_max_categories;
    std::vector<double> rf_priors;
    bool rf_calc_var_importance;
    int rf_n_active_vars;
    int rf_max_num_of_trees;
    double rf_accuracy;
  };

private:

  // Classifier parameters.
  Params params_;

  // Object to extract features from cluster points.
  FeatureExtractor feature_extractor_;

  // Actual random forest classifier.
  CvRTrees rtrees_;


public:

  RandomForestClassifier(const Params& params);

  /// \brief Classify clusters and return dynamic label.
  void classify(std::vector<Cluster>& cluster, std::vector<bool>* labels);

  /// \brief Load Random Forest from file.
  void load(const std::string& filename);

  /// \brief Save current Random Forest to file.
  void save(const std::string& filename) const;

  /// \brief Train Random Forest using features constructed from clusters and labels.
  void train(const std::vector<Cluster>& clusters, const std::vector<bool>& labels);

  /// \brief Test classifier and write ROC curve to file.
  void test(const std::vector<Cluster>& clusters, const std::vector<bool>& labels);
};

#endif /* OCTOMAP_COMPARE_RANDOM_FOREST_CLASSIFIER_H_ */
