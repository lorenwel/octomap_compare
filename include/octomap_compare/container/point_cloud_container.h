#ifndef POINT_CLOUD_CONTAINER_H_
#define POINT_CLOUD_CONTAINER_H_

#include "container_base.h"

#include <cmath>
#include <fstream>
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

  double max_x;
  double min_x;
  double max_y;
  double min_y;
  double max_z;
  double min_z;

  Eigen::Matrix3d std_dev_;
  Eigen::Matrix3d std_dev_inverse_;

  Eigen::Matrix3d spherical_transform_;

  std::vector<std::vector<double> > segmentation_;

  Matrix3xDynamic transformed_points_;

  Eigen::Vector3d sphericalToCartesian(const SphericalVector& spherical) const {
    const double x = spherical(2) * sin(spherical(1)) * cos(spherical(0));
    const double y = spherical(2) * sin(spherical(1)) * sin(spherical(0));
    const double z = spherical(2) * cos(spherical(1));
    return spherical_transform_.transpose() * Eigen::Vector3d(x, y, z);
  }

  inline std::pair<unsigned int, unsigned int>normalize(const unsigned int& phi_index,
                                                        const unsigned int& theta_index) const {
    return std::make_pair((phi_index + kNPhi) % kNPhi, (theta_index + kNTheta) % kNTheta);
  }

  std::pair<unsigned int, unsigned int> index(const SphericalPoint& point) const {
    const unsigned int phi_index = (point.phi + M_PI) * kPhiIndexFactor;
    const unsigned int theta_index = point.theta* kThetaIndexFactor;
    return normalize(phi_index, theta_index);
  }

  std::pair<unsigned int, unsigned int> index(const SphericalVector& point) const {
    const unsigned int phi_index = (point(0) + M_PI) * kPhiIndexFactor;
    const unsigned int theta_index = point(1) * kThetaIndexFactor;
    return normalize(phi_index, theta_index);
  }

  void updateBoundary(const Eigen::Vector3d& cur_point) {
    if (cur_point(0) > max_x) max_x = cur_point(0);
    else if (cur_point(0) < min_x) min_x = cur_point(0);
    if (cur_point(1) > max_y) max_y = cur_point(1);
    else if (cur_point(1) < min_y) min_y = cur_point(1);
    if (cur_point(2) > max_z) max_z = cur_point(2);
    else if (cur_point(2) < min_z) min_z = cur_point(2);
  }

//  void saveSegmentation() {
//    std::ofstream outfile;
//    outfile.open("/tmp/segmentation.csv");
//    if (outfile.is_open()) {
//      for (const auto& row: segmentation_) {
//        for (const auto& val: row) {
//          outfile << val << ", ";
//        }
//        outfile << "\n";
//      }
//      outfile.close();
//    }
//  }

  void processCloud() {
    const size_t n_points = occupied_points_.cols();
    SphericalPoint spherical ;
    for (size_t i = 0; i < n_points; ++i) {
      updateBoundary(occupied_points_.col(i));

      spherical = cartesianToSpherical(occupied_points_.col(i));
      spherical_points_.col(i) = SphericalVector(spherical);
      const auto ind = index(spherical);
      if (segmentation_[ind.first][ind.second] < spherical.r2) {
        segmentation_[ind.first][ind.second] = spherical.r2;
      }
    }
    const double offset = kNStdDev * std_dev_(2,2);
    max_x += offset; max_y += offset; max_z += offset;
    min_x -= offset; min_y -= offset; min_z -= offset;

    // Write Segmentation to file for debugging.
//    saveSegmentation();

    spherical_points_scaled_ = std_dev_inverse_ * spherical_points_;
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
    spherical_points_scaled_.resize(3, occupied_points_.cols());
    transformed_points_.resize(3, occupied_points_.cols());

    CHECK_EQ(spherical_points_.cols(),
             spherical_points_scaled_.cols());

    processCloud();

    // Create kd tree.
    kd_tree_ = std::unique_ptr<NNSearch3d>(
                   NNSearch3d::createKDTreeLinearHeap(spherical_points_scaled_));
    std::cout << "Created kd-tree\n";
  }

  SphericalPoint cartesianToSpherical(const Eigen::Vector3d& point_base) const {
    const Eigen::Vector3d point(spherical_transform_ * point_base);
    SphericalPoint spherical;
    spherical.r2 = point.squaredNorm();
    spherical.r = sqrt(spherical.r2);
    spherical.phi = atan2(point(1), point(0));
    spherical.theta = atan2(sqrt(point(0) * point(0) + point(1) * point(1)), point(2));
    return spherical;
  }

//  SphericalVector cartesianToSpherical(const Eigen::Vector3d& point) const {
//    const Eigen::Vector3d cartesian(spherical_transform_ * point);
//    const double phi = atan2(cartesian(1), cartesian(0));
//    const double theta = atan2(sqrt(cartesian(0) * cartesian(0) + cartesian(1) * cartesian(1)),
//                               cartesian(2));
//    const double r = cartesian.norm();
//    return SphericalVector(phi, theta, r);
//  }

  inline bool isInBBox(const Eigen::Vector3d& point) const {
    return (point(0) > min_x && point(0) < max_x &&
            point(1) > min_y && point(1) < max_y &&
            point(2) > min_z && point(2) < max_z);
  }

  bool isApproxObserved(const Eigen::Vector3d& point,
                        SphericalPoint* spherical_coordinates,
                        SphericalPoint* spherical_coordinates_scaled) const {
    // Check BBox first so we can void a lot of atan2.
    if (!isInBBox(point)) return false;
    // Check spherical coordinates.
    *spherical_coordinates = cartesianToSpherical(point);
    *spherical_coordinates_scaled = std_dev_inverse_ * Eigen::Vector3d(*spherical_coordinates);

    return true;
  }

  bool isObserved(const SphericalPoint& spherical_point, const double dist_correction = 0) const {
    // Find relevant segmentation indices.
    const int upper_limit_phi = dist_correction * std_dev_(0,0) * kPhiIndexFactor;
    const int lower_limit_phi = -upper_limit_phi;
    const int upper_limit_theta = dist_correction * std_dev_(1,1) * kThetaIndexFactor;
    const int lower_limit_theta = -upper_limit_theta;
    auto ind = index(spherical_point);
//    std::cout << "phi " << upper_limit_phi << " theta " << upper_limit_theta << " corr " << dist_correction << "\n";
    std::pair<std::vector<int>, std::vector<int> > ind_bounds;
    for (int i = lower_limit_phi; i <= upper_limit_phi; ++i) {
      ind_bounds.first.push_back(ind.first + i);
    }
    for (int i = lower_limit_theta; i <= upper_limit_theta; ++i) {
      ind_bounds.second.push_back(ind.second + i);
    }
    for (auto it = ind_bounds.first.begin(); *it < 0; ++it) {
      *it += kNPhi;
    }
    for (auto it = ind_bounds.second.begin(); *it < 0; ++it) {
      *it += kNPhi;
    }
    for (auto it = ind_bounds.first.rbegin(); *it >= kNPhi; ++it) {
      *it -= kNPhi;
    }
    for (auto it = ind_bounds.second.rbegin(); *it >= kNTheta; ++it) {
      *it -= kNPhi;
    }
    // Check all.
    for (const auto& phi_index: ind_bounds.first) {
      for (const auto& theta_index: ind_bounds.second) {
        if (segmentation_[phi_index][theta_index] > spherical_point.r2) return true;
      }
    }
    return false;
  }

  inline Matrix3xDynamic& TransformedPoints() {
    return transformed_points_;
  }

  inline const Matrix3xDynamic& TransformedPoints() const {
    return transformed_points_;
  }

};

#endif /* POINT_CLOUD_CONTAINER_H_ */
