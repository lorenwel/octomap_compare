#ifndef POINT_CLOUD_CONTAINER_H_
#define POINT_CLOUD_CONTAINER_H_

#include "container_base.h"

#include <cmath>
#include <limits>

static unsigned int kNPhi = 2880;
static unsigned int kNTheta = 2880;
static const double kPhiIndexFactor = kNPhi / (2.0*M_PI);
static const double kThetaIndexFactor = kNTheta / M_PI;

class PointCloudContainer : public ContainerBase {

  struct SphericalPoint {
    double r2, phi, theta;
  };

  double max_x;
  double min_x;
  double max_y;
  double min_y;
  double max_z;
  double min_z;

  std::vector<std::vector<double>> segmentation_;

  SphericalPoint cartesianToSpherical(const Eigen::Vector3d& point) const {
    SphericalPoint spherical;
    spherical.r2 = point.squaredNorm();
    spherical.phi = atan2(point(1), point(0));
    spherical.theta = atan2(sqrt(point(0) * point(0) + point(1) * point(1)), point(2));
    return spherical;
  }

  std::pair<unsigned int, unsigned int> index(const SphericalPoint& point) const {
    const unsigned int phi_index = (point.phi + M_PI) * kPhiIndexFactor;
    const unsigned int theta_index = point.theta* kThetaIndexFactor;
    return std::make_pair(phi_index, theta_index);
  }

  void processCloud() {
    for (size_t i = 0; i < occupied_points_.cols(); ++i) {
      const Eigen::Vector3d cur_point = occupied_points_.col(i);
      if (cur_point(0) > max_x) max_x = cur_point(0);
      else if (cur_point(0) < min_x) min_x = cur_point(0);
      if (cur_point(1) > max_y) max_y = cur_point(1);
      else if (cur_point(1) < min_y) min_y = cur_point(1);
      if (cur_point(2) > max_z) max_z = cur_point(2);
      else if (cur_point(2) < min_z) min_z = cur_point(2);

      const SphericalPoint spherical = cartesianToSpherical(cur_point);
      const auto ind = index(spherical);
      if (segmentation_[ind.first][ind.second] < spherical.r2)
        segmentation_[ind.first][ind.second] = spherical.r2;
    }
  }

 public:
  PointCloudContainer(const Eigen::MatrixXd& points) :
      segmentation_(kNPhi, std::vector<double>(kNTheta, 0)) {
    occupied_points_ = points;
    // Create kd tree.
    kd_tree_ = std::unique_ptr<Nabo::NNSearchD>(
        Nabo::NNSearchD::createKDTreeLinearHeap(occupied_points_));
    std::cout << "Created kd-tree\n";

    processCloud();
  }

  inline bool isInBBox(const Eigen::Vector3d& point) const {
    return (point(0) > min_x && point(0) < max_x &&
            point(1) > min_y && point(1) < max_y &&
            point(2) > min_z && point(2) < max_z);
  }

  bool isObserved(const Eigen::Vector3d& point) const {
    // Check BBox first so we can void a lot of atan2.
    if (!isInBBox(point)) return false;
    // Check spherical coordinates.
    const SphericalPoint spherical = cartesianToSpherical(point);
    const auto ind = index(spherical);
    const double r2 = point.squaredNorm();
    return (segmentation_[ind.first][ind.second] > r2);
  }

};

#endif /* POINT_CLOUD_CONTAINER_H_ */
