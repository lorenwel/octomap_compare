#ifndef CONTAINER_BASE_H_
#define CONTAINER_BASE_H_

#include <memory>

#include <nabo/nabo.h>


class ContainerBase {
public:
  struct KNNResult {
    // 3D position of closest neighbors.
    Eigen::Matrix<double, 3, Eigen::Dynamic> points;
    // Distances of closest neighbors.
    Eigen::VectorXd distances;
    // Indices of closest points.
    Eigen::VectorXi indices;

    KNNResult(const unsigned int& n_neighbors) {
      points.resize(3, n_neighbors);
      distances.resize(n_neighbors);
      indices.resize(n_neighbors);
    }
  };

protected:

  // Eigen Matrix with occupied points.
  Eigen::MatrixXd occupied_points_;

  Eigen::MatrixXd spherical_points_;

  // kd-tree for nearest neighbor search.
  std::unique_ptr<Nabo::NNSearchD> kd_tree_;

public:

  /// \brief Find n nearest neighbors of point.
  KNNResult findKNN(const Eigen::Vector3d& point, const unsigned int& n_neighbors) const {
    // Query.
    KNNResult result(n_neighbors);
    kd_tree_->knn(point, result.indices, result.distances,
                  n_neighbors, 0, Nabo::NNSearchD::SORT_RESULTS);

    //Fill QueryResult.
    for (unsigned int i = 0; i < n_neighbors; ++i) {
      const int cur_index = result.indices(i);
      result.points.col(i) = occupied_points_.col(cur_index);
    }
    return result;
  }

  /// \brief Access to point cloud matrix.
  inline const Eigen::MatrixXd& Points() const {
    return occupied_points_;
  }

  /// \brief Access to point cloud matrix.
  inline const Eigen::MatrixXd& SphericalPoints() const {
    return spherical_points_;
  }

};

#endif // CONTAINER_BASE_H_
