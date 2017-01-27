#ifndef OCTOMAP_COMPARE_UTILS_H_
#define OCTOMAP_COMPARE_UTILS_H_

#include <list>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pointmatcher/PointMatcher.h>
#include <ros/ros.h>

#include <sm/timing/Timer.hpp>

// Typedefs for timing code.
// Don't activeate both timers at the same time or the pipeline timer will be false.
// Times subsets of pipeline.
//typedef sm::timing::Timer Timer;
typedef sm::timing::DummyTimer Timer;
// Times the whole pipeline.
//typedef sm::timing::Timer PipelineTimer;
typedef sm::timing::DummyTimer PipelineTimer;

typedef PointMatcher<double> PM;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Matrix3xDynamic;
typedef Nabo::NearestNeighbourSearch<double, Matrix3xDynamic> NNSearch3d;
typedef Eigen::Vector3d SphericalVector;
typedef std::vector<std::pair<int, Eigen::Vector3d>> ClusterCentroidVector;

struct SphericalPoint {
  double r2, phi, theta, r;

  SphericalPoint() = default;
  SphericalPoint(const Eigen::Vector3d& in) {
    phi = in(0);
    theta = in(1);
    r = in(2);
    r2 = r*r;
  }

  SphericalPoint& operator= (const Eigen::Vector3d& in) {
    phi = in(0);
    theta = in(1);
    r = in(2);
    r2 = r*r;

    return *this;
  }

  operator Eigen::Vector3d () const {
    return Eigen::Vector3d(phi, theta, r);
  }
};

struct ClusterPoint {
  Eigen::Vector3d cartesian;
  Eigen::Vector3d spherical;
  double distance;
};

struct Cluster {
  int id;
  float score;
  std::vector<ClusterPoint> points;

  Cluster(const int& id_) : id(id_) {}
};

static void applyColorToPoint(const pcl::RGB& color, pcl::PointXYZRGB& point) {
  point.r = color.r;
  point.g = color.g;
  point.b = color.b;
}

template <typename PointType>
static void setXYZFromEigen(const Eigen::Vector3d& eigen, PointType& point) {
  point.x = eigen(0);
  point.y = eigen(1);
  point.z = eigen(2);
}

static void clusterToPointCloud(const Cluster& cluster,
                                pcl::PointCloud<pcl::PointXYZRGB>* cloud,
                                const Eigen::Affine3d& transform = Eigen::Affine3d::Identity()) {
  CHECK_NOTNULL(cloud)->clear();
  pcl::PointXYZRGB point;
  pcl::RGB color;
  if (cluster.id > 0) {color.r = 0; color.g = 255; color.b = 0;}
  else if (cluster.id < 0) {color.r = 255; color.g = 0; color.b = 0;}
  else {color.r = 180; color.g = 180; color.b = 180;}
  for (const ClusterPoint& cluster_point: cluster.points) {
    setXYZFromEigen(transform * cluster_point.cartesian, point);
    applyColorToPoint(color, point);
    cloud->push_back(point);
  }
}

inline pcl::RGB getColorFromIdSign(const int& id) {
  pcl::RGB color;
  if (id < 0) {color.r = 255; color.g = 0; color.b = 0;}
  else if (id > 0) {color.r = 0; color.g = 255; color.b = 0;}
  else {color.r = 180; color.g = 180; color.b = 180;}
  return color;
}

static size_t sumOfEqualValues(const std::vector<bool>& vec1, const std::vector<bool>& vec2,
                        const bool& val1, const bool& val2,
                               const std::vector<size_t> n_points) {
  CHECK_EQ(vec1.size(), vec2.size()) << "Vectors have different length.";
  const size_t vec_size = vec1.size();
  size_t sum = 0;
  for (size_t i = 0; i < vec_size; ++i) {
    if (vec1[i] == val1 && vec2[i] == val2) sum += n_points[i];
  }
  LOG_IF(WARNING, sum == 0) << val1 << " " << val2 << " has sum 0.";
  return sum;
}

static size_t sumOfEqualValues(const std::vector<bool>& vec, const bool& val,
                               const std::vector<size_t> n_points) {
  const size_t vec_size = vec.size();
  size_t sum = 0;
  for (size_t i = 0; i < vec_size; ++i) {
    if (vec[i] == val) sum += n_points[i];
  }
  LOG_IF(WARNING, sum == 0) << val << " has sum 0.";
  return sum;
}

static std::pair<float, float> getFalsePosAndTruePos(
    const std::vector<bool>& pred_labels,
    const std::vector<bool>& true_labels,
    const std::vector<size_t> n_points) {
  CHECK_EQ(pred_labels.size(), true_labels.size()) << "Vectors have different length.";
  CHECK_EQ(pred_labels.size(), n_points.size()) << "Vectors have different length.";
  const double n_true_pos = sumOfEqualValues(pred_labels, true_labels, true, true, n_points);
  const double n_true_neg = sumOfEqualValues(pred_labels, true_labels, false, false, n_points);
  const double n_false_pos = sumOfEqualValues(pred_labels, true_labels, true, false, n_points);
  const double n_false_neg = sumOfEqualValues(pred_labels, true_labels, false, true, n_points);
  return std::make_pair(n_false_pos / (n_false_pos + n_true_neg), // FalsePos.
                        n_true_pos / (n_true_pos + n_false_neg));  // TruePos.
}

inline std::pair<float, float> getFalsePosAndTruePos(
    const std::vector<bool>& pred_labels,
    const std::vector<bool>& true_labels) {
  return getFalsePosAndTruePos(pred_labels,
                               true_labels,
                               std::vector<size_t>(pred_labels.size(), 1));
}



static std::pair<float, float> getPrecisionAndRecall(
    const std::vector<bool>& pred_labels,
    const std::vector<bool>& true_labels,
    const std::vector<size_t> n_points) {
  CHECK_EQ(pred_labels.size(), true_labels.size()) << "Vectors have different length.";
  CHECK_EQ(pred_labels.size(), n_points.size()) << "Vectors have different length.";
  const double n_pos = sumOfEqualValues(true_labels, true, n_points);
  const double n_neg = sumOfEqualValues(true_labels, false, n_points);
  const double n_true_pos = sumOfEqualValues(pred_labels, true_labels, true, true, n_points);
  const double n_false_pos = sumOfEqualValues(pred_labels, true_labels, true, false, n_points);
  const double n_false_neg = sumOfEqualValues(pred_labels, true_labels, false, true, n_points);
  return std::make_pair(n_true_pos/n_pos / (n_true_pos/n_pos + n_false_pos/n_neg),  // Precision.
                        n_true_pos / (n_true_pos + n_false_neg)); // Recall.
}

inline std::pair<float, float> getPrecisionAndRecall(
    const std::vector<bool>& pred_labels,
    const std::vector<bool>& true_labels) {
  return getPrecisionAndRecall(pred_labels,
                               true_labels,
                               std::vector<size_t>(pred_labels.size(), 1));
}

inline double getNextVal(std::stringstream& stream) {
  CHECK(stream.good()) << "Stream is broken, can't read next value.";
  std::string val;
  std::getline(stream, val, ',');
  return std::stof(val);
}

static bool parseCsvToCluster(const std::string& file_name,
                              std::vector<Cluster>* clusters,
                              std::vector<bool>* labels) {
  CHECK_NOTNULL(clusters)->clear();
  CHECK_NOTNULL(labels)->clear();
  std::ifstream file;
  std::list<std::pair<ClusterPoint, int> > points_with_id;
  ClusterPoint point;
  int id;

  file.open(file_name);
  if (!file.is_open()) {
    LOG(ERROR) << "Could not open file: " << file_name;
    return false;
  }
  LOG(INFO) << "Opened file \"" << file_name << "\".";

  std::string line;
  std::stringstream stream;
  std::unordered_set<int> dynamic_clusters;
  // Go through lines until we get values.
  while (file.peek() == 'x') {
    std::getline(file, line);
  }
  while (std::getline(file, line)) {
    // Check if we reached end of file and should read dynamic cluster.
    if (line.size() <= 1) {
      std::getline(file, line);
      stream.clear();
      stream.str(line);
      std::string val;
      while (std::getline(stream, val, ',')) {
        LOG(INFO) << "Trying to read cluster from " << val << ".";
        // Check if this was the trailing comma.
        if (val != " ") dynamic_clusters.insert(std::stod(val));
      }
      // Terminate loop.
      break;
    }
    // Extract line.
    stream.clear();
    stream.str(line);
    point.cartesian(0) = getNextVal(stream);
    point.cartesian(1) = getNextVal(stream);
    point.cartesian(2) = getNextVal(stream);
    point.spherical(0) = getNextVal(stream);
    point.spherical(1) = getNextVal(stream);
    point.spherical(2) = getNextVal(stream);
    point.distance = getNextVal(stream);
    // TODO: Check if the following really works. It should but is probabliy inefficient;
    id = getNextVal(stream);
    points_with_id.push_back(std::make_pair(point, id));

//    LOG(INFO) << "Line read " << point.cartesian(0) << " "
//                              << point.cartesian(1) << " "
//                              << point.cartesian(2) << " "
//                              << point.spherical(0) << " "
//                              << point.spherical(1) << " "
//                              << point.spherical(2) << " "
//                              << point.distance << " " << id << ".";
  }
  file.close();

  // Find all cluster ids.
  std::unordered_set<int> cluster_ids;
  for (const auto& point_id: points_with_id) {
    cluster_ids.insert(point_id.second);
  }

  // Create clusters.
  std::unordered_map<int, size_t> id_to_ind;
  for (const int& cur_id: cluster_ids) {
    id_to_ind[cur_id] = clusters->size();
    clusters->push_back(Cluster(cur_id));
    if (dynamic_clusters.count(cur_id)) labels->push_back(true);
    else labels->push_back(false);
  }
  CHECK_EQ(clusters->size(), labels->size()) << "Labels and clusters dont't have same size.";

  // Fill clusters.
  for (const auto& point_id: points_with_id) {
    clusters->at(id_to_ind[point_id.second]).points.push_back(point_id.first);
  }

  // Print cluster sizes.
  for (const auto& cluster: *clusters) {
    LOG(INFO) << "Cluster " << cluster.id << " has " << cluster.points.size() << " points.";
  }

  return true;
}

static double getCompareDist(const Eigen::VectorXd& distances2,
                             const std::string& dist_metric = "max",
                             const double& correction = 0) {
  double dist;
  if (dist_metric == "max") {
    dist = sqrt(distances2(distances2.size() - 1));
  }
  else if (dist_metric == "mean") {
    dist = sqrt(distances2.mean());
  }
  else if (dist_metric == "min") {
    dist = sqrt(distances2(0));
  }
  else {
    LOG(FATAL) << "Invalid distances metric \"" << dist_metric << "\".\n";
    return -1;
  }
  dist -= correction;
  if (dist < 0) dist = 0;
  return dist;
}

/// \brief Converts a 3xN Eigen matrix to a PointMatcher cloud.
static PM::DataPoints matrix3dEigenToPointMatcher(const Matrix3xDynamic& matrix) {
  PM::DataPoints::Labels xyz1;
  xyz1.push_back(PM::DataPoints::Label("x", 1));
  xyz1.push_back(PM::DataPoints::Label("y", 1));
  xyz1.push_back(PM::DataPoints::Label("z", 1));
  xyz1.push_back(PM::DataPoints::Label("pad", 1));
  PM::DataPoints point_matcher(xyz1, PM::DataPoints::Labels(), matrix.cols());
  point_matcher.getFeatureViewByName("x") = matrix.row(0);
  point_matcher.getFeatureViewByName("y") = matrix.row(1);
  point_matcher.getFeatureViewByName("z") = matrix.row(2);
  point_matcher.getFeatureViewByName("pad").setConstant(1);

  return point_matcher;
}

/// \brief Converts a 3xN Eigen matrix to a PointMatcher cloud.
static Matrix3xDynamic pointMatcherToMatrix3dEigen(const PM::DataPoints& points) {
  const size_t n_points = points.getNbPoints();
  Matrix3xDynamic matrix(3, n_points);
  matrix.row(0) = points.getFeatureViewByName("x");
  matrix.row(1) = points.getFeatureViewByName("y");
  matrix.row(2) = points.getFeatureViewByName("z");

  return matrix;
}

static void matrixToPointCloud(const Matrix3xDynamic matrix,
                        pcl::PointCloud<pcl::PointXYZ>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  for (unsigned int i = 0; i < matrix.cols(); ++i) {
    pcl::PointXYZ point(matrix(0,i), matrix(1,i), matrix(2,i));
    cloud->push_back(point);
  }
}

/// \brief Gets point color based on distance for use in visualization.
static pcl::RGB getColorFromDistance(const double& dist, double max_dist = 1) {
  if (max_dist < 1) max_dist = 1; // Avoid crazy heat map with small max_dist.
  const double first = max_dist / 4;
  const double second = max_dist / 2;
  const double third = first + second;
  pcl::RGB color;
  if (dist > max_dist) {
    color.r = 255;
    color.g = 0;
    color.b = 0;
  }
  else if (dist > third) {
    color.r = 255;
    color.g = (max_dist - dist)/first*255;
    color.b = 0;
  }
  else if (dist > second) {
    color.r = (dist - second)/first*255;
    color.g = 255;
    color.b = 0;
  }
  else if (dist > first) {
    color.r = 0;
    color.g = 255;
    color.b = (second - dist)/first*255;
  }
  else if (dist >= 0) {
    color.r = 0;
    color.g = dist/first * 255;
    color.b = 255;
  }
  else {
    // Negative distance. Used to mark previously unobserved points.
    color.r = 180;
    color.g = 180;
    color.b = 180;
  }
  return color;
}

#endif /* OCTOMAP_COMPARE_UTILS_H_ */
