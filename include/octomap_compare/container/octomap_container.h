#ifndef OCTOMAP_CONTAINER_H_
#define OCTOMAP_CONTAINER_H_

#include "container_base.h"

#include <unordered_map>

#include <octomap/OcTree.h>

class OctomapContainer : public ContainerBase {

  // Octomap.
  std::shared_ptr<octomap::OcTree> octree_;

  // Robot frame observed points.
  Matrix3xDynamic transformed_points_;

  /// \brief Processes octree to enable comparison.
  void processTree();

public:
  /// \brief Read octomap from file.
  OctomapContainer(const std::string& file);

  /// \brief Use octree passed to constructor.
  OctomapContainer(const std::shared_ptr<octomap::OcTree>& octree);

  /// \brief Checks if voxel at point was observed. Also passes back pointer to node at point.
  bool isObserved(const Eigen::Vector3d& point, octomap::OcTreeNode** node) const;

  /// \brief Set observed points transformed into spherical coordinates.
  ///        Scaled with sensor uncertainty.
  void setSphericalScaled(const std::list<SphericalPoint>& spherical_points_scaled) {
    const size_t n_spherical = spherical_points_scaled.size();
    spherical_points_scaled_.resize(3, n_spherical);
    size_t counter = 0;
    for (auto&& point : spherical_points_scaled) {
      spherical_points_scaled_.col(counter++) = (Eigen::Vector3d)point;
    }
    if (n_spherical > 0) {
      kd_tree_ = std::unique_ptr<NNSearch3d>(
          NNSearch3d::createKDTreeLinearHeap(spherical_points_scaled_));
    }
  }

  /// \brief Set observed points transformed into spherical coordinates.
  void setSpherical(const std::list<SphericalPoint>& spherical_points) {
    const size_t n_spherical = spherical_points.size();
    spherical_points_.resize(3, n_spherical);
    size_t counter = 0;
    for (auto&& point : spherical_points) {
      spherical_points_.col(counter++) = (Eigen::Vector3d)point;
    }
  }

  /// \brief Set observed points transformed into robot frame.
  void setTransformedPoints(const std::list<Eigen::Vector3d>& points) {
    const size_t n_points = points.size();
    transformed_points_.resize(3, n_points);
    size_t counter = 0;
    for (auto&& point: points) {
      transformed_points_.col(counter++) = point;
    }
  }

  /// \brief Operator overload to allow access to underlying octree.
  std::shared_ptr<octomap::OcTree> operator->() const {
    return octree_;
  }

  /// \brief Used to check if OcTree is loaded.
  inline operator bool() const {
    return (bool)octree_;
  }

  /// \brief Returns transformed points.
  inline const Matrix3xDynamic& TransformedPoints() const {
    return transformed_points_;
  }

};

#endif // OCTOMAP_CONTAINER_H_
