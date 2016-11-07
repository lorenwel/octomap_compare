#ifndef POINT_CLOUD_CONTAINER_H_
#define POINT_CLOUD_CONTAINER_H_

#include "container_base.h"

class PointCloudContainer : public ContainerBase {

 public:
  PointCloudContainer(const Eigen::MatrixXd& points) {
    occupied_points_ = points;
    // Create kd tree.
    kd_tree_ = std::unique_ptr<Nabo::NNSearchD>(
        Nabo::NNSearchD::createKDTreeLinearHeap(occupied_points_));
    std::cout << "Created kd-tree\n";
  }

};

#endif /* POINT_CLOUD_CONTAINER_H_ */
