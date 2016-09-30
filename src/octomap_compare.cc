#include "octomap_compare/octomap_compare.h"

#include <list>
#include <limits>

#include <dbscan/dbscan.h>
#include <pcl/conversions.h>

OctomapCompare::OctomapCompare(const std::string& base_file, const std::string& comp_file,
                               const CompareParams& params = CompareParams()) :
    base_octree_(base_file),
    comp_octree_(comp_file),
    params_(params) {}

OctomapCompare::CompareResult OctomapCompare::compare() {
  CompareResult result;
  if (!base_octree_ || !comp_octree_) {
    std::cerr << "Not all OcTrees defined\n";
    return result;
  }
  std::cout << "Comparing...\n";
  // TODO: Change n to a parameter.
  const unsigned int n = 5; // Number of nearest neighbors.
  // Do the comparison.
  std::list<Eigen::Vector3d> observed_points;
  std::list<Eigen::Vector3d> unobserved_points;
  std::list<Eigen::VectorXd> distances;
  double min = std::numeric_limits<double>::max();
  double max = 0;
  for (auto it = comp_octree_->begin_leafs(); it != comp_octree_->end_leafs(); ++it) {
    if (comp_octree_->isNodeOccupied(*it)) {
      Eigen::Vector3d query_point(it.getX(), it.getY(), it.getZ());
      double dist = 0;
      if (base_octree_.isObserved(query_point)) {
        observed_points.push_back(query_point);
        OctomapContainer::KNNResult knn_result = base_octree_.findKNN(query_point, n);
        distances.push_back(knn_result.distances);
        dist = knn_result.distances[0];
        if (dist > max) max = dist;
        if (dist < min) min = dist;
      }
      else {
        unobserved_points.push_back(query_point);
      }
    }
  }
  // Fill result matrices.
  const unsigned int observed_size = observed_points.size();
  const unsigned int unobserved_size = unobserved_points.size();
  result.observed_points.resize(3, observed_size);
  result.unobserved_points.resize(3, unobserved_size);
  result.distances.resize(n, observed_size);
  result.max_dist = max;
  for (unsigned int i = 0; i < observed_size; ++i) {
    result.observed_points.col(i) = observed_points.front();
    observed_points.pop_front();
    result.distances.col(i) = distances.front();
    distances.pop_front();
  }
  for (unsigned int i = 0; i < unobserved_size; ++i) {
    result.unobserved_points.col(i) = unobserved_points.front();
    unobserved_points.pop_front();
  }
  std::cout << "Compare cloud has " << observed_size << " observed and "
                                    << unobserved_size << " unobserved voxels.\n";
  std::cout << "min dist: " << sqrt(min) << "\n";
  std::cout << "max dist: " << sqrt(max) << "\n";
  return result;
}

void OctomapCompare::getChanges(const CompareResult& result,
                                Eigen::Matrix<double, 3, Eigen::Dynamic>* output,
                                Eigen::VectorXi* cluster) {
  CHECK_NOTNULL(output);
  // Apply threshold.
  std::list<unsigned int> indices;
  const double thres = params_.distance_threshold * params_.distance_threshold;
  for (unsigned int i = 0; i < result.distances.cols(); ++i) {
    if (result.distances(0,i) > thres) indices.push_back(i);
  }
  output->resize(3, indices.size());
  unsigned int counter = 0;
  for (auto&& index : indices) {
    output->col(counter++) = result.observed_points.col(index);
  }
  std::cout << output->cols() << " voxels left after distance thresholding\n";
  // DBSCAN filtering.
  Dbscan dbscan(output->transpose(), params_.eps, params_.min_pts);  
  dbscan.cluster(cluster);

}

void OctomapCompare::compareResultToPointCloud(const CompareResult& result,
                                               pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  const double max_dist = sqrt(result.max_dist);
  for (unsigned int i = 0; i < result.observed_points.cols(); ++i) {
    // Write point to cloud.
    pcl::RGB color = getColorFromDistance(sqrt(result.distances(0,i)), max_dist);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.observed_points(0,i);
    pointrgb.y = result.observed_points(1,i);
    pointrgb.z = result.observed_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
  for (unsigned int i = 0; i < result.unobserved_points.cols(); ++i) {
    // Write point to cloud.
    pcl::RGB color = getColorFromDistance(-1.0);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.unobserved_points(0,i);
    pointrgb.y = result.unobserved_points(1,i);
    pointrgb.z = result.unobserved_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
}

pcl::RGB OctomapCompare::getColorFromDistance(const double& dist, const double& max_dist) {
  const double first = max_dist / 4;
  const double second = max_dist / 2;
  const double third = first + second;
  pcl::RGB color;
  if (dist > max_dist) {
    color.r = 255;
    color.g = 0;
    color.b = 0;
  }
  else if (dist > third) {
    color.r = 255;
    color.g = (max_dist - dist)/first*255;
    color.b = 0;
  }
  else if (dist > second) {
    color.r = (dist - second)/first*255;
    color.g = 255;
    color.b = 0;
  }
  else if (dist > first) {
    color.r = 0;
    color.g = 255;
    color.b = (second - dist)/first*255;
  }
  else if (dist > 0) {
    color.r = 0;
    color.g = dist/first * 255;
    color.b = 255;
  }
  else {
    // Negative distance. Used to mark previously unobserved points.
    color.r = 180;
    color.g = 180;
    color.b = 180;
  }
  return color;
}
