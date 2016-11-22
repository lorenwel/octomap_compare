#ifndef POINT_CLOUD_CONTAINER_H_
#define POINT_CLOUD_CONTAINER_H_

#include "container_base.h"

#include <cmath>
#include <limits>

static unsigned int kNPhi = 2880;
static unsigned int kNTheta = 1440;
static const double kPhiIndexFactor = kNPhi / (2.0*M_PI);
static const double kThetaIndexFactor = kNTheta / M_PI;

static const unsigned int kNStdDev = 2;

typedef Eigen::Vector3d SphericalVector;

class PointCloudContainer : public ContainerBase {

  struct SphericalPoint {
    double r, r2, phi, theta;
  };

  double max_x;
  double min_x;
  double max_y;
  double min_y;
  double max_z;
  double min_z;

  Eigen::Matrix3d std_dev_;
  Eigen::Matrix3d std_dev_inverse_;

  Eigen::Matrix3d spherical_transform_;

  std::vector<std::vector<double>> segmentation_;

  SphericalPoint cartesianToSpherical(const Eigen::Vector3d& point_base) const {
    const Eigen::Vector3d point(spherical_transform_ * point_base);
    SphericalPoint spherical;
    spherical.r2 = point.squaredNorm();
    spherical.r = sqrt(spherical.r2);
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
    const unsigned int phi_offset = kNStdDev * std_dev_(0,0) * kPhiIndexFactor;
    const unsigned int theta_offset = kNStdDev * std_dev_(1,1) * kThetaIndexFactor;

    const size_t n_points = occupied_points_.cols();
    for (size_t i = 0; i < n_points; ++i) {
      const Eigen::Vector3d cur_point = occupied_points_.col(i);
      if (cur_point(0) > max_x) max_x = cur_point(0);
      else if (cur_point(0) < min_x) min_x = cur_point(0);
      if (cur_point(1) > max_y) max_y = cur_point(1);
      else if (cur_point(1) < min_y) min_y = cur_point(1);
      if (cur_point(2) > max_z) max_z = cur_point(2);
      else if (cur_point(2) < min_z) min_z = cur_point(2);

      const SphericalPoint spherical = cartesianToSpherical(cur_point);
      spherical_points_.col(i) = Eigen::Vector3d(spherical.phi,
                                                 spherical.theta,
                                                 spherical.r);
      const auto ind = index(spherical);
      // Add kNPhi/Theta to avoid negative indices so we can do modulo for circular index.
      const std::pair<unsigned int, unsigned int> phi_bound(ind.first - phi_offset + kNPhi,
                                                            ind.first + phi_offset + kNPhi);
      const std::pair<unsigned int, unsigned int> theta_bound(ind.second - theta_offset + kNTheta,
                                                              ind.second + theta_offset + kNTheta);
      for (unsigned int phi_index = phi_bound.first; phi_index <= phi_bound.second; ++phi_index) {
        for (unsigned int theta_index = theta_bound.first;
             theta_index <= theta_bound.second; ++theta_index) {
          const unsigned int i_phi = phi_index % kNPhi;
          const unsigned int i_theta = theta_index % kNTheta;
          if (segmentation_[i_phi][i_theta] < spherical.r2)
            segmentation_[i_phi][i_theta] = spherical.r2;
        }
      }
    }
    static const double offset = kNStdDev * std_dev_(2,2);
    max_x += offset; max_y += offset; max_z += offset;
    min_x -= offset; min_y -= offset; min_z -= offset;

    spherical_points_ = std_dev_inverse_ * spherical_points_;
  }

 public:
  PointCloudContainer(const Eigen::MatrixXd& points,
                      const Eigen::Matrix3d& spherical_transform,
                      const Eigen::Matrix3d& std_dev) :
      segmentation_(kNPhi, std::vector<double>(kNTheta, 0)),
      std_dev_(std_dev),
      spherical_transform_(spherical_transform) {
    std_dev_inverse_ = std_dev_.inverse();
    occupied_points_ = points;
    spherical_points_.resize(3, occupied_points_.cols());

    processCloud();

    // Create kd tree.
    kd_tree_ = std::unique_ptr<Nabo::NNSearchD>(
        Nabo::NNSearchD::createKDTreeLinearHeap(spherical_points_));
    std::cout << "Created kd-tree\n";

  }

  inline bool isInBBox(const Eigen::Vector3d& point) const {
    return (point(0) > min_x && point(0) < max_x &&
            point(1) > min_y && point(1) < max_y &&
            point(2) > min_z && point(2) < max_z);
  }

  bool isObserved(const Eigen::Vector3d& point, SphericalVector* spherical_coordinates) const {
    // Check BBox first so we can void a lot of atan2.
    if (!isInBBox(point)) return false;
    // Check spherical coordinates.
    const SphericalPoint spherical = cartesianToSpherical(point);
    const SphericalVector out(spherical.phi,
                              spherical.theta,
                              spherical.r);
    *spherical_coordinates = std_dev_inverse_ * out;

    const auto ind = index(spherical);
    const double r2 = point.squaredNorm();
    return (segmentation_[ind.first][ind.second] > r2);
  }

};

#endif /* POINT_CLOUD_CONTAINER_H_ */
