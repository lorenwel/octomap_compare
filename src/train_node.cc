#include "octomap_compare/random_forest_classifier.h"

#include <boost/filesystem.hpp>
#include <ros/ros.h>

typedef boost::filesystem::directory_iterator BoostDirIter;

void readAllCsvFilesInDirectory(const std::string& dir,
                                std::vector<Cluster>* clusters,
                                std::vector<bool>* labels) {
  boost::filesystem::path path(dir);
  CHECK(boost::filesystem::is_directory(path)) << dir << " is not a directory.";
  LOG(INFO) << "Reading directory " << path << ".";

  std::vector<Cluster> temp_clusters;
  std::vector<bool> temp_labels;
  for (const auto& file: boost::make_iterator_range(BoostDirIter(path), {})) {
    if (file.path().extension() == ".csv") {
      const std::string file_name = file.path().string();
      if (parseCsvToCluster(file_name, &temp_clusters, &temp_labels)) {
        clusters->insert(clusters->end(), temp_clusters.begin(), temp_clusters.end());
        labels->insert(labels->end(), temp_labels.begin(), temp_labels.end());
        LOG(INFO) << "Successfully parsed file " << file.path() << ".";
      }
      else {
        LOG(ERROR) << "Failed to parse file " << file.path() << ".";
      }
    }
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "train_node");
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 3) << "Wrong number of command line arguments.";

  ros::NodeHandle nh("~");

  RandomForestClassifier::Params params;

  nh.param("classifier/histogram/min_val", params.histogram_params.min_val, params.histogram_params.min_val);
  int n_bins(params.histogram_params.n_bins);
  nh.param("classifier/histogram/n_bins", n_bins, n_bins);
  params.histogram_params.n_bins = n_bins;
  nh.param("classifier/histogram/bin_size", params.histogram_params.bin_size,
                                 params.histogram_params.bin_size);

  nh.param("classifier/probability_threshold", params.probability_threshold, params.probability_threshold);
  nh.param("classifier/save_classifier_after_training", params.save_classifier_after_training,
                                             params.save_classifier_after_training);
  nh.param("classifier_file_name", params.classifier_file_name, params.classifier_file_name);
  nh.param("classifier/roc_filename", params.roc_filename, params.roc_filename);

  nh.param("classifier/rf_max_depth", params.rf_max_depth, params.rf_max_depth);
  nh.param("classifier/rf_min_sample_ratio", params.rf_min_sample_ratio, params.rf_min_sample_ratio);
  nh.param("classifier/rf_priors", params.rf_priors, params.rf_priors);
  nh.param("classifier/rf_calc_var_importance", params.rf_calc_var_importance, params.rf_calc_var_importance);
  nh.param("classifier/rf_use_surrogates", params.rf_use_surrogates, params.rf_use_surrogates);
  nh.param("classifier/rf_n_active_vars", params.rf_n_active_vars, params.rf_n_active_vars);
  nh.param("classifier/rf_max_num_of_trees", params.rf_max_num_of_trees, params.rf_max_num_of_trees);
  nh.param("classifier/rf_accuracy", params.rf_accuracy, params.rf_accuracy);
  // The following are not actually used by the random tree. Need to be set regardless.
  nh.param("classifier/rf_regression_accuracy", params.rf_regression_accuracy, params.rf_regression_accuracy);
  nh.param("classifier/rf_max_categories", params.rf_max_categories, params.rf_max_categories);


  RandomForestClassifier classifier(params);

  // Read data.
  std::vector<Cluster> train_data, test_data;
  std::vector<bool> train_labels, test_labels;
  std::string train_directory(argv[1]), test_directory(argv[2]);
  readAllCsvFilesInDirectory(train_directory, &train_data, &train_labels);
  readAllCsvFilesInDirectory(test_directory, &test_data, &test_labels);

  // Train.
  classifier.train(train_data, train_labels);

  LOG(INFO) << "Finished training.";

  // Test.
  classifier.test(test_data, test_labels);
  LOG(INFO) << "Finished testing.";


}
