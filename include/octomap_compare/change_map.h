#ifndef OCTOMAP_COMPARE_CHANGE_MAP_H_
#define OCTOMAP_COMPARE_CHANGE_MAP_H_

#include <memory>

#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>

template<typename PointType>
class ChangeMap {
  typedef pcl::PointCloud<PointType> CloudType;
  typedef typename pcl::PointCloud<PointType>::ConstPtr CloudTypeConstPtr;
  typedef pcl::octree::OctreePointCloudPointVector<PointType> PclOcTree;

  CloudType change_map_;

  std::unique_ptr<PclOcTree> last_octree_;
  std::unique_ptr<PclOcTree> input_octree_;
  PclOcTree valid_octree_;

  CloudType change_points_;

  const double max_val_;

public:
  ChangeMap(const double resolution) : last_octree_(new PclOcTree(resolution)),
                                       input_octree_(new PclOcTree(resolution)),
                                       valid_octree_(resolution),
                                       max_val_(resolution * 32768) {
    change_points_.header.frame_id = "map";
    last_octree_->defineBoundingBox(-max_val_, -max_val_, -max_val_, max_val_, max_val_, max_val_);
    input_octree_->defineBoundingBox(-max_val_, -max_val_, -max_val_, max_val_, max_val_, max_val_);
    valid_octree_.defineBoundingBox(-max_val_, -max_val_, -max_val_, max_val_, max_val_, max_val_);
  }

  void addPointCloud(const CloudTypeConstPtr& input_cloud) {
    input_octree_->setInputCloud(input_cloud);
    input_octree_->addPointsFromInputCloud();
    if (last_octree_->getInputCloud()) {
      for (auto it = last_octree_->leaf_begin(); it != last_octree_->leaf_end(); ++it) {
        const std::vector<int> point_indices = it.getLeafContainer().getPointIndicesVector();
        const pcl::octree::OctreeKey key = it.getCurrentOctreeKey();
        // Check if input_octree_ hast same occupied leaf as last_octree_.
        if (point_indices.size() > 0) {
          const auto* ptr = input_octree_->findLeaf(key.x, key.y, key.z);
          if (ptr) {
            LOG(INFO) << "At depth " << it.getCurrentOctreeDepth();
            LOG(INFO) << "Found other non-empty voxel.";
            if (ptr->getSize() > 0) {
              LOG(INFO) << "Found other non-empty voxel that has points.";
              // Input_octree_ has same leaf --> add leaf to valid_octree_.
              valid_octree_.createLeaf(key.x, key.y, key.z);
            }
          }
        }
        // Check if points in this last_octree_ leaf are valid.
        if (valid_octree_.existLeaf(key.x, key.y, key.z)) {
          // Add them if they are.
          change_points_ += CloudType(*(last_octree_->getInputCloud()), point_indices);
          LOG(INFO) << "Added point.";
        }
      }
    }
    double min_x, max_x, min_y, max_y, min_z, max_z;
    input_octree_->getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    LOG(INFO) << "input bounding box " << min_x<< " " << min_y<< " " << min_z<< " " << max_x<< " " << max_y << " " << max_z;
    last_octree_->getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    LOG(INFO) << "last bounding box " << min_x<< " " << min_y<< " " << min_z<< " " << max_x<< " " << max_y << " " << max_z;
    valid_octree_.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    LOG(INFO) << "valid bounding box " << min_x<< " " << min_y<< " " << min_z<< " " << max_x<< " " << max_y << " " << max_z;
    LOG(INFO) << "input tree depth " << input_octree_->getTreeDepth();
    LOG(INFO) << "last tree depth " << last_octree_->getTreeDepth();
    LOG(INFO) << "valid tree depth " << valid_octree_.getTreeDepth();

    // Prepare for next pass.
    last_octree_.swap(input_octree_);
    input_octree_->deleteTree();
    input_octree_->defineBoundingBox(-max_val_, -max_val_, -max_val_, max_val_, max_val_, max_val_);
  }

  inline const CloudType& getCloud() {
    return change_points_;
  }
};

#endif /* OCTOMAP_COMPARE_CHANGE_MAP_H_ */
