#ifndef OCTOMAP_COMPARE_CHANGE_MAP_H_
#define OCTOMAP_COMPARE_CHANGE_MAP_H_

#include <algorithm>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "octomap_compare/octomap_compare_utils.h"

struct Change {
  Matrix3xDynamic points;
  Eigen::AlignedBox3d bbox;
  float score;
};

template<typename PointType>
class ChangeMap {

  typedef pcl::PointCloud<PointType> CloudType;
  typedef typename CloudType::Ptr CloudTypePtr;
  typedef typename CloudType::ConstPtr CloudTypeConstPtr;

  const double min_ratio_;

  const unsigned int min_pts_;

  std::vector<Change> appear_changes_;
  std::vector<Change> disappear_changes_;

public:
  ChangeMap(const double& resolution, const unsigned int& min_pts) :
            min_ratio_(resolution),
            min_pts_(min_pts) {

  }

  void addPointCloud(const std::vector<Cluster>& clusters,
                     const std::vector<bool>& labels,
                     const Eigen::Affine3d& transform = Eigen::Affine3d::Identity()) {
    CHECK_EQ(clusters.size(), labels.size());
    Change temp_change;
    const size_t n_clusters = clusters.size();
    // Check if we have dynamic cluster.
    if (std::find(labels.begin(), labels.end(), true) != labels.end()) {
      for (size_t i = 0; i < n_clusters; ++i) {
        if (labels[i]) {
          const size_t n_points = clusters[i].points.size();
          temp_change.points.resize(3, n_points);
          for (size_t j = 0; j < n_points; ++j) {
            temp_change.points.col(j) = transform * clusters[i].points[j].cartesian;
          }
          const Eigen::Vector3d min = temp_change.points.rowwise().minCoeff();
          const Eigen::Vector3d max = temp_change.points.rowwise().maxCoeff();
          temp_change.bbox = Eigen::AlignedBox3d(min, max);
          temp_change.score = clusters[i].score;
          if (clusters[i].id > 0) appear_changes_.push_back(temp_change);
          else if (clusters[i].id < 0) disappear_changes_.push_back(temp_change);
        }
      }
    }
  }

  // TODO: make this more efficient.
  std::list<Eigen::AlignedBox3d> getBBoxIntersection(const std::vector<Change>& changes) {
    std::list<Eigen::AlignedBox3d> bbox_list;

    const size_t n_scans = changes.size();
    size_t counter;
    Eigen::AlignedBox3d cur_box;
    for (size_t i = 0; i < n_scans; ++i) {
      cur_box = changes[i].bbox;
      counter = 1;
      for (size_t j = i+1; j < n_scans; ++j) {
        const Eigen::AlignedBox3d intersection = cur_box.intersection(changes[j].bbox);
        if (!intersection.isEmpty()){
          const double intersection_volume = intersection.volume();
          if (intersection_volume / cur_box.volume() > (1-changes[i].score) &&
              intersection_volume / changes[j].bbox.volume() > (1-changes[j].score)) {
//            cur_box = intersection;
            ++counter;
            if (counter >= min_pts_) {
              bbox_list.push_back(cur_box);
              break;
            }
          }
        }
      }
    }

    return bbox_list;
  }

  void addPointsInOverlappingBBoxToCloud(const std::vector<Change>& changes,
                                         const std::list<Eigen::AlignedBox3d>& bboxes,
                                         const pcl::RGB& color,
                                         CloudTypePtr& cloud) {
    PointType point;
    applyColorToPoint(color, point);
    for (const Change& change: changes) {
      for (const Eigen::AlignedBox3d& bbox: bboxes) {
        if (change.bbox.intersects(bbox)) {
          if (change.bbox.intersection(bbox).volume() / change.bbox.volume() > (1-change.score)) {
            const size_t n_points = change.points.cols();
            for (size_t i = 0; i < n_points; ++i) {
              setXYZFromEigen(change.points.col(i), point);
              cloud->push_back(point);
            }
            break;
          }
        }
      }
    }
  }

  CloudTypeConstPtr getCloud() {
    CloudTypePtr out_cloud(new CloudType());
    CloudTypePtr temp_cloud(new CloudType());

    // Build point cloud.
    const std::list<Eigen::AlignedBox3d> appear_bboxes = getBBoxIntersection(appear_changes_);
    const std::list<Eigen::AlignedBox3d> disappear_bboxes =
        getBBoxIntersection(disappear_changes_);
    pcl::RGB color; color.r = 0; color.g = 255; color.b = 0;
    addPointsInOverlappingBBoxToCloud(appear_changes_, appear_bboxes, color, temp_cloud);
    color.r = 255; color.g = 0;
    addPointsInOverlappingBBoxToCloud(disappear_changes_, disappear_bboxes, color, temp_cloud);

    // Filter.
    pcl::StatisticalOutlierRemoval<PointType> sor_filter;
    sor_filter.setInputCloud(temp_cloud);
    sor_filter.setMeanK(10);
    sor_filter.setStddevMulThresh(2.0);
    sor_filter.filter(*out_cloud);

    out_cloud->header.frame_id = "map";
    return out_cloud;
  }
};

#endif /* OCTOMAP_COMPARE_CHANGE_MAP_H_ */
