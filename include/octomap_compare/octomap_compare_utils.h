#ifndef OCTOMAP_COMPARE_UTILS_H_
#define OCTOMAP_COMPARE_UTILS_H_

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pointmatcher/PointMatcher.h>

#include "octomap_compare/octomap_container.h"

typedef PointMatcher<double> PM;

double getCompareDist(const Eigen::VectorXd& distances, const std::string& dist_metric = "max") {
  if (dist_metric == "max") return distances(distances.size() - 1);
  else if (dist_metric == "mean") return distances.mean();
  else if (dist_metric == "min") return distances(0);
  else {
    std::cerr << "Invalid distances metric.\n";
    return -1;
  }
}

/// \brief Creates new OctomapContainer object with Octomap created from arg.
template <typename T>
void loadOctomap(const T& arg, std::shared_ptr<OctomapContainer>* container) {
  *container = std::make_shared<OctomapContainer>(arg);
}

/// \brief Converts a 3xN Eigen matrix to a PointMatcher cloud.
PM::DataPoints matrix3dEigenToPointMatcher(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& matrix) {
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

void extractCluster(const Eigen::MatrixXd& points, const Eigen::VectorXi& cluster_indices,
                    std::vector<Eigen::MatrixXd>* cluster, Eigen::MatrixXd* outlier) {

}

void changesToPointCloud(const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix,
                         const Eigen::VectorXi& cluster,
                         pcl::PointCloud<pcl::PointXYZRGB>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  std::srand(0);
  std::unordered_map<unsigned int, pcl::RGB> color_map;
  std::unordered_set<int> cluster_indices;
  for (unsigned int i = 0; i < cluster.rows(); ++i) {
    cluster_indices.insert(cluster(i));
    pcl::RGB color;
    color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
    color_map[cluster(i)] = color;
  }
  std::cout << "Found " << cluster_indices.size() - 1 << " clusters\n";
  for (unsigned int i = 0; i < matrix.cols(); ++i) {
    if (cluster(i) != 0) {
      pcl::RGB color = color_map[cluster(i)];
      pcl::PointXYZRGB point(color.r, color.g, color.b);
      point.x = matrix(0,i);
      point.y = matrix(1,i);
      point.z = matrix(2,i);
      cloud->push_back(point);
    }
  }
  std::cout << cloud->size() << " points left after clustering\n";
}

void matrixToPointCloud(const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix,
                        pcl::PointCloud<pcl::PointXYZ>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  for (unsigned int i = 0; i < matrix.cols(); ++i) {
    pcl::PointXYZ point(matrix(0,i), matrix(1,i), matrix(2,i));
    cloud->push_back(point);
  }
}

#endif /* OCTOMAP_COMPARE_UTILS_H_ */
