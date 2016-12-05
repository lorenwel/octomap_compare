#ifndef POINT_CLOUD_CONTAINER_H_
#define POINT_CLOUD_CONTAINER_H_

#include "container_base.h"

#include <cmath>
#include <limits>

#include <sm/timing/Timer.hpp>

//typedef sm::timing::Timer Timer;
typedef sm::timing::DummyTimer Timer;

static constexpr unsigned int kNPhi = 2880;
static constexpr unsigned int kNTheta = 1440;
static constexpr double kPhiIndexFactor = kNPhi / (2.0*M_PI);
static constexpr double kThetaIndexFactor = kNTheta / M_PI;

static constexpr unsigned int kNStdDev = 2;

class PointCloudContainer : public ContainerBase {

  struct SphericalPointR2 {
    double r2, phi, theta;
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

  Matrix3xDynamic transformed_points_;

  SphericalPointR2 cartesianToSphericalR2(const Eigen::Vector3d& point_base) const {
    const Eigen::Vector3d point(spherical_transform_ * point_base);
    SphericalPointR2 spherical;
    spherical.r2 = point.squaredNorm();
    spherical.phi = atan2(point(1), point(0));
    spherical.theta = atan2(sqrt(point(0) * point(0) + point(1) * point(1)), point(2));
    return spherical;
  }

  Eigen::Vector3d sphericalToCartesian(const SphericalVector& spherical) const {
    const double x = spherical(2) * sin(spherical(1)) * cos(spherical(0));
    const double y = spherical(2) * sin(spherical(1)) * sin(spherical(0));
    const double z = spherical(2) * cos(spherical(1));
    return spherical_transform_.transpose() * Eigen::Vector3d(x, y, z);
  }

  std::pair<unsigned int, unsigned int> index(const SphericalPointR2& point) const {
    const unsigned int phi_index = (point.phi + M_PI) * kPhiIndexFactor;
    const unsigned int theta_index = point.theta* kThetaIndexFactor;
    return std::make_pair(phi_index, theta_index);
  }

  std::pair<unsigned int, unsigned int> index(const SphericalVector& point) const {
    const unsigned int phi_index = (point(0) + M_PI) * kPhiIndexFactor;
    const unsigned int theta_index = point(1) * kThetaIndexFactor;
    return std::make_pair(phi_index, theta_index);
  }

  void updateBoundary(const Eigen::Vector3d& cur_point) {
    if (cur_point(0) > max_x) max_x = cur_point(0);
    else if (cur_point(0) < min_x) min_x = cur_point(0);
    if (cur_point(1) > max_y) max_y = cur_point(1);
    else if (cur_point(1) < min_y) min_y = cur_point(1);
    if (cur_point(2) > max_z) max_z = cur_point(2);
    else if (cur_point(2) < min_z) min_z = cur_point(2);
  }

  void processCloud() {
    const unsigned int phi_offset = kNStdDev * std_dev_(0,0) * kPhiIndexFactor;
    const unsigned int theta_offset = kNStdDev * std_dev_(1,1) * kThetaIndexFactor;

    const size_t n_points = occupied_points_.cols();
    for (size_t i = 0; i < n_points; ++i) {
      Timer boundary_timer("UpdateBoundary");
      updateBoundary(occupied_points_.col(i));
      boundary_timer.stop();

      Timer cart_to_spherical("CartToSpherical");
      const SphericalPointR2 spherical = cartesianToSphericalR2(occupied_points_.col(i));
      cart_to_spherical.stop();
      Timer sqrt_timer("Sqrt");
      spherical_points_.col(i) = SphericalVector(spherical.phi,
                                                 spherical.theta,
                                                 sqrt(spherical.r2));
      sqrt_timer.stop();
      Timer get_index("GetIndex");
      const auto ind = index(spherical);
      get_index.stop();
      // Add kNPhi/Theta to avoid negative indices so we can do modulo for circular index.
      Timer allocate_pair("Allocate");
      const std::pair<unsigned int, unsigned int> phi_bound(ind.first - phi_offset + kNPhi,
                                                            ind.first + phi_offset + kNPhi);
      const std::pair<unsigned int, unsigned int> theta_bound(ind.second - theta_offset + kNTheta,
                                                              ind.second + theta_offset + kNTheta);
      allocate_pair.stop();
      Timer update_segmentation("UpdateSegmentation");
//      Timer mod_timer("Mod");
//      mod_timer.stop();
//      Timer comp_timer("Comp");
//      comp_timer.stop();
      std::cout << "phi_bound " << phi_bound.first << " " << phi_bound.second
                << "theta_bound " << theta_bound.first << " " << theta_bound.second << "\n";
      for (unsigned int phi_index = phi_bound.first; phi_index <= phi_bound.second; ++phi_index) {
        for (unsigned int theta_index = theta_bound.first;
             theta_index <= theta_bound.second; ++theta_index) {
//          mod_timer.start();
          const unsigned int i_phi = phi_index % kNPhi;
          const unsigned int i_theta = theta_index % kNTheta;
//          mod_timer.stop();
//          comp_timer.start();
          if (segmentation_[i_phi][i_theta] < spherical.r2) {
            segmentation_[i_phi][i_theta] = spherical.r2;
          }
//          comp_timer.stop();
        }
      }
      update_segmentation.stop();
    }
    static const double offset = kNStdDev * std_dev_(2,2);
    max_x += offset; max_y += offset; max_z += offset;
    min_x -= offset; min_y -= offset; min_z -= offset;

    spherical_points_ = std_dev_inverse_ * spherical_points_;
  }

 public:
  PointCloudContainer(const Matrix3xDynamic& points,
                      const Eigen::Matrix3d& spherical_transform,
                      const Eigen::Matrix3d& std_dev) :
      segmentation_(kNPhi, std::vector<double>(kNTheta, 0)),
      std_dev_(std_dev),
      spherical_transform_(spherical_transform) {
    std_dev_inverse_ = std_dev_.inverse();
    occupied_points_ = points;
    spherical_points_.resize(3, occupied_points_.cols());
    transformed_points_.resize(3, occupied_points_.cols());

    processCloud();

    // Create kd tree.
    kd_tree_ = std::unique_ptr<NNSearch3d>(
                   NNSearch3d::createKDTreeLinearHeap(spherical_points_));
    std::cout << "Created kd-tree\n";

  }

  SphericalVector cartesianToSpherical(const Eigen::Vector3d& point) const {
    const Eigen::Vector3d cartesian(spherical_transform_ * point);
    const double phi = atan2(cartesian(1), cartesian(0));
    const double theta = atan2(sqrt(cartesian(0) * cartesian(0) + cartesian(1) * cartesian(1)),
                               cartesian(2));
    const double r = cartesian.norm();
    return SphericalVector(phi, theta, r);
  }

  inline bool isInBBox(const Eigen::Vector3d& point) const {
    return (point(0) > min_x && point(0) < max_x &&
            point(1) > min_y && point(1) < max_y &&
            point(2) > min_z && point(2) < max_z);
  }

  bool isApproxObserved(const Eigen::Vector3d& point, SphericalVector* spherical_coordinates) const {
    // Check BBox first so we can void a lot of atan2.
    if (!isInBBox(point)) return false;
    // Check spherical coordinates.
    const SphericalVector out = cartesianToSpherical(point);
    *spherical_coordinates = std_dev_inverse_ * out;

    const auto ind = index(out);
    const double r2 = point.squaredNorm();
    return (segmentation_[ind.first][ind.second] > r2);
  }

  bool isObserved(const SphericalVector& spherical_point) const {
    auto ind = index(spherical_point);
    ind.first = (ind.first + kNPhi) % kNPhi;
    ind.second = (ind.second + kNTheta) % kNTheta;
    const double r2 = spherical_point(2) * spherical_point(2);
    return (segmentation_[ind.first][ind.second] > r2);
  }

  inline Matrix3xDynamic& TransformedPoints() {
    return transformed_points_;
  }

  inline const Matrix3xDynamic& TransformedPoints() const {
    return transformed_points_;
  }


};

#endif /* POINT_CLOUD_CONTAINER_H_ */
