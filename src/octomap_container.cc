#include "octomap_compare/octomap_container.h"

#include <cassert>

OctomapContainer::OctomapContainer(const std::string& file) {
  std::cout << "Reading octomap from " << file << "\n";
  octree_ = std::make_shared<octomap::OcTree>(file);

  processTree();
}

OctomapContainer::OctomapContainer(const std::shared_ptr<octomap::OcTree>& octree)
  : octree_(octree) {
  processTree();
}

OctomapContainer::KNNResult OctomapContainer::findKNN(const Eigen::Vector3d& point,
                                                       const unsigned int& n_neighbors) const {
  // Query.
  Eigen::VectorXi indices(n_neighbors);
  Eigen::VectorXd distances(n_neighbors);
  kd_tree_->knn(point, indices, distances, n_neighbors, 0, Nabo::NNSearchD::SORT_RESULTS);

  //Fill QueryResult.
  KNNResult result(n_neighbors);
  result.distances = distances;
  for (unsigned int i = 0; i < n_neighbors; ++i) {
    const unsigned int cur_index = indices[i];
    result.keys[i] = key_map_[cur_index];
    result.points.col(i) = occupied_points_.col(cur_index);
  }
  return result;
}

bool OctomapContainer::isObserved(const Eigen::Vector3d &point, octomap::OcTreeNode** node) const {
  *node = octree_->search(point.x(), point.y(), point.z());
  return *node;
}

void OctomapContainer::processTree() {
  const unsigned int max_tree_depth = octree_->getTreeDepth();
  const double resolution = octree_->getResolution();

  // Get number of occupied voxels.
  unsigned int voxel_count = 0;
  for (octomap::OcTree::leaf_iterator it = octree_->begin_leafs();
       it != octree_->end_leafs(); ++it) {
    if (octree_->isNodeOccupied(*it)) {
      const int depth_diff = max_tree_depth - it.getDepth();
      voxel_count += pow(2, depth_diff * 3);
    }
  }

  std::cout << "Found " << voxel_count << " occupied voxels\n";
  occupied_points_.resize(3, voxel_count);
  key_map_.resize(voxel_count);

  // Fill matrix.
  unsigned int index = 0;
  for (octomap::OcTree::leaf_iterator it = octree_->begin_leafs();
       it != octree_->end_leafs(); ++it) {
    if (octree_->isNodeOccupied(*it)) {
      if (max_tree_depth == it.getDepth()) {
        occupied_points_.col(index) = Eigen::Vector3d(it.getX(), it.getY(), it.getZ());
        key_map_[index++] = it.getKey();
      }
      // If leaf is not max depth it represents an occupied voxel with edge
      // length of 2^(max_tree_depth - leaf_depth) * resolution.
      // We use multiple points to visualize this filled volume.
      else {
        const unsigned int box_edge_length = pow(2, max_tree_depth - it.getDepth() - 1);
        const double bbx_offset = box_edge_length * resolution - resolution/2;
        Eigen::Vector3d bbx_offset_vec(bbx_offset, bbx_offset, bbx_offset);
        Eigen::Vector3d center(it.getX(), it.getY(), it.getZ());
        Eigen::Vector3d bbx_min = center - bbx_offset_vec;
        Eigen::Vector3d bbx_max = center + bbx_offset_vec;
        // Add small offset to avoid overshooting bbx_max.
        bbx_max += Eigen::Vector3d::Constant(0.001);
        // Save index before these loops to check proper execution when debugging.
        for (double x_position = bbx_min.x(); x_position <= bbx_max.x();
             x_position += resolution) {
          for (double y_position = bbx_min.y(); y_position <= bbx_max.y();
               y_position += resolution) {
            for (double z_position = bbx_min.z(); z_position <= bbx_max.z();
                 z_position += resolution) {
              occupied_points_.col(index) = Eigen::Vector3d(x_position, y_position, z_position);
              key_map_[index++] = it.getKey();
            }
          }
        }
      }
    }
  }
  if (index != voxel_count) std::cerr << "End index does not match voxel count\n";
  else std::cout << "Successfully created point matrix\n";

  // Create kd tree.
  kd_tree_ = std::unique_ptr<Nabo::NNSearchD>(
      Nabo::NNSearchD::createKDTreeLinearHeap(occupied_points_));
  std::cout << "Created kd-tree\n";

}
