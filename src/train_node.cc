#include "octomap_compare/load_parameters.h"
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

  RandomForestClassifier::Params params = getRandomForestParams(nh);

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
