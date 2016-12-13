#ifndef CONTAINER_BASE_H_
#define CONTAINER_BASE_H_

#include <memory>

#include <nabo/nabo.h>

#include "octomap_compare/octomap_compare_utils.h"

class ContainerBase {
public:
  struct KNNResult {
    // Distances of closest neighbors.
    Eigen::VectorXd distances2;
    // Indices of closest points.
    Eigen::VectorXi indices;

    KNNResult(const unsigned int& n_neighbors) {
      distances2.resize(n_neighbors);
      indices.resize(n_neighbors);
    }
  };

protected:

  // Eigen Matrix with occupied points.
  Matrix3xDynamic occupied_points_;

  Matrix3xDynamic spherical_points_;
  Matrix3xDynamic spherical_points_scaled_;

  // kd-tree for nearest neighbor search.
  std::unique_ptr<NNSearch3d> kd_tree_;

public:

  /// \brief Find n nearest neighbors of point.
  KNNResult findKNN(const Eigen::Vector3d& point, const unsigned int& n_neighbors) const {
    // Query.
    KNNResult result(n_neighbors);
    kd_tree_->knn(point, result.indices, result.distances2,
                  n_neighbors, 0, NNSearch3d::SORT_RESULTS);
    return result;
  }

  /// \brief Access to point cloud matrix.
  inline const Matrix3xDynamic& Points() const {
    return occupied_points_;
  }

  /// \brief Access to spherical coordinate point cloud matrix.
  inline const Matrix3xDynamic& SphericalPoints() const {
    return spherical_points_;
  }

  /// \brief Access to spherical coordinate point cloud matrix. ATTENTION! THESE ARE SCALED WITH THE INVERSE OF STD_DEV
  inline const Matrix3xDynamic& SphericalPointsScaled() const {
    return spherical_points_scaled_;
  }

};

#endif // CONTAINER_BASE_H_
