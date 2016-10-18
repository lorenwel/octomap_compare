#ifndef OCTOMAP_CONTAINER_H_
#define OCTOMAP_CONTAINER_H_

#include <memory>

#include <nabo/nabo.h>
#include <octomap/OcTree.h>


class OctomapContainer {
public:
  struct KNNResult {
    // 3D position of closest neighbors.
    Eigen::Matrix<double, 3, Eigen::Dynamic> points;
    // Distances of closest neighbors.
    Eigen::VectorXd distances;
    // Keys of closest neighbor voxels
    std::vector<octomap::OcTreeKey> keys;

    KNNResult(const unsigned int& n_neighbors) {
      points.resize(3, n_neighbors);
      keys.resize(n_neighbors);
    }
  };

private:

  // Octomap.
  std::shared_ptr<octomap::OcTree> octree_;

  // Eigen Matrix with occupied voxels of octree_.
//  Eigen::Matrix<double, 3, Eigen::Dynamic> occupied_points; // libnabo doesn't like this.
  Eigen::MatrixXd occupied_points_;

  // Maps index index of point in occupied_points
  // to key of corresponding voxel in octree_.
  std::vector<octomap::OcTreeKey> key_map_;

  // kd-tree for nearest neighbor search.
  std::unique_ptr<Nabo::NNSearchD> kd_tree_;

  /// \brief Processes octree to enable comparison.
  void processTree();

public:
  /// \brief Read octomap from file.
  OctomapContainer(const std::string& file);

  /// \brief Use octree passed to constructor.
  OctomapContainer(const std::shared_ptr<octomap::OcTree>& octree);

  /// \brief Find n nearest neighbors of point.
  KNNResult findKNN(const Eigen::Vector3d& point, const unsigned int& n_neighbors) const;

  /// \brief Checks if voxel at point was observed. Also passes back pointer to node at point.
  bool isObserved(const Eigen::Vector3d& point, octomap::OcTreeNode** node) const;

  /// \brief Operator overload to allow access to underlying octree.
  std::shared_ptr<octomap::OcTree> operator->() {
    return octree_;
  }

  /// \brief Used to check if OcTree is loaded.
  operator bool() const {
    return (bool)octree_;
  }

  const Eigen::MatrixXd& Points() const {
    return occupied_points_;
  }

};

#endif // OCTOMAP_CONTAINER_H_
