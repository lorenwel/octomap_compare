#ifndef OCTOMAP_COMPARE_UTILS_H_
#define OCTOMAP_COMPARE_UTILS_H_

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pointmatcher/PointMatcher.h>

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
