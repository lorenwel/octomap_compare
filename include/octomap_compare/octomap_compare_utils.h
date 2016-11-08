#ifndef OCTOMAP_COMPARE_UTILS_H_
#define OCTOMAP_COMPARE_UTILS_H_

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pointmatcher/PointMatcher.h>

#include "octomap_compare/container/octomap_container.h"

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

/// \brief Converts a 3xN Eigen matrix to a PointMatcher cloud.
Eigen::Matrix<double, 3, Eigen::Dynamic> pointMatcherToMatrix3dEigen(
    const PM::DataPoints& points) {
  const size_t n_points = points.getNbPoints();
  Eigen::Matrix<double, 3, Eigen::Dynamic> matrix(3, n_points);
  matrix.row(0) = points.getFeatureViewByName("x");
  matrix.row(1) = points.getFeatureViewByName("y");
  matrix.row(2) = points.getFeatureViewByName("z");

  return matrix;
}

void extractCluster(const Eigen::MatrixXd& points, const Eigen::VectorXi& cluster_indices,
                    std::vector<Eigen::MatrixXd>* cluster, Eigen::MatrixXd* outlier) {

}

void changesToPointCloud(const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix_appear,
                         const Eigen::Matrix<double, 3, Eigen::Dynamic> matrix_disappear,
                         const Eigen::VectorXi& cluster_appear,
                         const Eigen::VectorXi& cluster_disappear,
                         const bool& color_changes,
                         pcl::PointCloud<pcl::PointXYZRGB>* cloud) {
  CHECK_NOTNULL(cloud)->clear();
  cloud->header.frame_id = "map";
  std::srand(0);
  std::unordered_map<unsigned int, pcl::RGB> color_map_appear;
  std::unordered_map<unsigned int, pcl::RGB> color_map_disappear;
  std::unordered_set<int> cluster_indices;
  // Generate cluster to color map.
  if (!color_changes) {
    for (unsigned int i = 0; i < cluster_appear.rows(); ++i) {
      cluster_indices.insert(cluster_appear(i));
      pcl::RGB color;
      color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
      color_map_appear[cluster_appear(i)] = color;
    }
    cluster_indices.clear();
    for (unsigned int i = 0; i < cluster_disappear.rows(); ++i) {
      cluster_indices.insert(cluster_disappear(i));
      pcl::RGB color;
      color.r = rand() * 255; color.g = rand() * 255; color.b = rand() * 255;
      color_map_disappear[cluster_disappear(i)] = color;
    }
  }
  // Build point cloud.
  for (unsigned int i = 0; i < matrix_appear.cols(); ++i) {
    if (cluster_appear(i) != 0) {
      pcl::RGB color;
      if (color_changes) {
        color.r = 0; color.g = 255; color.b = 0;
      }
      else {
        color = color_map_appear[cluster_appear(i)];
      }
      pcl::PointXYZRGB point(color.r, color.g, color.b);
      point.x = matrix_appear(0,i);
      point.y = matrix_appear(1,i);
      point.z = matrix_appear(2,i);
      cloud->push_back(point);
    }
  }
  for (unsigned int i = 0; i < matrix_disappear.cols(); ++i) {
    if (cluster_disappear(i) != 0) {
      pcl::RGB color;
      if (color_changes) {
        color.r = 255; color.g = 0; color.b = 0;
      }
      else {
        color = color_map_disappear[cluster_disappear(i)];
      }
      pcl::PointXYZRGB point(color.r, color.g, color.b);
      point.x = matrix_disappear(0,i);
      point.y = matrix_disappear(1,i);
      point.z = matrix_disappear(2,i);
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

/// \brief Gets point color based on distance for use in visualization.
pcl::RGB getColorFromDistance(const double& dist, double max_dist = 1) {
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
