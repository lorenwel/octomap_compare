#include "octomap_compare/container/octomap_container.h"

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

bool OctomapContainer::isObserved(const Eigen::Vector3d &point, octomap::OcTreeNode** node) const {
  *node = octree_->search(point.x(), point.y(), point.z());
  return *node;
}

void OctomapContainer::processTree() {
  const unsigned int max_tree_depth = octree_->getTreeDepth();
  const double resolution = octree_->getResolution();

  // TODO: Put this in main loop. Iterating through leafs is slow.
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

  // Fill matrix.
  size_t index = 0;
  for (octomap::OcTree::leaf_iterator it = octree_->begin_leafs();
       it != octree_->end_leafs(); ++it) {
    if (octree_->isNodeOccupied(*it)) {
      if (max_tree_depth == it.getDepth()) {
        occupied_points_.col(index) = Eigen::Vector3d(it.getX(), it.getY(), it.getZ());
        ++index;
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
              ++index;
            }
          }
        }
      }
    }
  }
  if (index != voxel_count) std::cerr << "End index does not match voxel count\n";
  else std::cout << "Successfully created point matrix\n";
}
