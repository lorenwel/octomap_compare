#include "octomap_compare/octomap_compare.h"

#include <list>
#include <limits>
#include <thread>

#include <dbscan/dbscan.h>
#include <pcl/conversions.h>

#include "octomap_compare/octomap_compare_utils.h"

OctomapCompare::OctomapCompare(const std::string& base_file, const std::string& comp_file,
                               const CompareParams& params) :
    params_(params) {
  std::thread base_thread(loadOctomap<std::string>, base_file, &base_octree_);
  std::thread comp_thread(loadOctomap<std::string>, comp_file, &comp_octree_);
  base_thread.join();
  comp_thread.join();
}

OctomapCompare::OctomapCompare(const std::shared_ptr<octomap::OcTree>& base_tree,
                               const std::shared_ptr<octomap::OcTree>& comp_tree,
                               const CompareParams &params) :
    params_(params) {
  std::thread base_thread(loadOctomap<std::shared_ptr<octomap::OcTree>>, base_tree, &base_octree_);
  std::thread comp_thread(loadOctomap<std::shared_ptr<octomap::OcTree>>, comp_tree, &comp_octree_);
  base_thread.join();
  comp_thread.join();
}

OctomapCompare::CompareResult OctomapCompare::compare() {
  CompareResult result;
  if (!base_octree_ || !comp_octree_) {
    std::cerr << "Not all OcTrees defined\n";
    return result;
  }
  std::cout << "Comparing...\n";

  // Do ICP.
  getTransformFromICP();

  std::list<Eigen::Vector3d> comp_observed_points;
  std::list<Eigen::VectorXd> comp_distances;
  std::list<Eigen::Vector3d> unobserved_points;
  std::list<octomap::OcTreeKey> base_keys;
  double max_dist =
      compareForward(&comp_observed_points, &comp_distances, &unobserved_points, &base_keys);

  // Create map that matches an octree key to the distance to its closest comp neighbor.
  KeyToDistMap key_to_dist;
  if (params_.k_nearest_neighbor == 1) {
    for (auto iter_pair = std::make_pair(base_keys.begin(), comp_distances.begin());
         iter_pair.first != base_keys.end() && iter_pair.second != comp_distances.end();
         ++iter_pair.first, ++iter_pair.second) {
      const double cur_dist = (*iter_pair.second)(0);
      auto key_iter = key_to_dist.emplace(*iter_pair.first, cur_dist);
      // Do comparison if element already exists.
      if (!key_iter.second && key_iter.first->second > cur_dist) {
        key_iter.first->second = cur_dist;
      }
    }
  }

  std::list<Eigen::Vector3d> base_observed_points;
  std::list<Eigen::VectorXd> base_distances;
  compareBackward(key_to_dist, max_dist, &base_observed_points, &base_distances, &unobserved_points);

  // Fill result matrices.
  const size_t base_size = base_observed_points.size();
  const size_t comp_size = comp_observed_points.size();
  const size_t unobserved_size = unobserved_points.size();
  result.base_observed_points.resize(3, base_size);
  result.comp_observed_points.resize(3, comp_size);
  result.unobserved_points.resize(3, unobserved_size);
  result.base_distances.resize(params_.k_nearest_neighbor, base_size);
  result.comp_distances.resize(params_.k_nearest_neighbor, comp_size);
  result.max_dist = max_dist;
  for (size_t i = 0; i < base_size; ++i) {
    result.base_observed_points.col(i) = base_observed_points.front();
    base_observed_points.pop_front();
    result.base_distances.col(i) = base_distances.front();
    base_distances.pop_front();
  }
  for (size_t i = 0; i < comp_size; ++i) {
    result.comp_observed_points.col(i) = comp_observed_points.front();
    comp_observed_points.pop_front();
    result.comp_distances.col(i) = comp_distances.front();
    comp_distances.pop_front();
  }
  for (size_t i = 0; i < unobserved_size; ++i) {
    result.unobserved_points.col(i) = unobserved_points.front();
    unobserved_points.pop_front();
  }
  std::cout << "Compare cloud has " << comp_size << " observed voxels\n";
  std::cout << "Base cloud has " << base_size << " observed voxels\n";
  std::cout << "There are a combined " << unobserved_size << " unobserved voxels.\n";
  std::cout << "max dist: " << sqrt(max_dist) << "\n";

  return result;
}

double OctomapCompare::compareForward(std::list<Eigen::Vector3d>* observed_points,
                                      std::list<Eigen::VectorXd>* distances,
                                      std::list<Eigen::Vector3d>* unobserved_points,
                                      std::list<octomap::OcTreeKey>* keys) {
  CHECK_NOTNULL(observed_points);
  CHECK_NOTNULL(distances);
  CHECK_NOTNULL(unobserved_points);
  CHECK_NOTNULL(keys);
  double max_dist = 0;
  // Do the comparison.
  for (auto it = (*comp_octree_)->begin_leafs(); it != (*comp_octree_)->end_leafs(); ++it) {
    if ((*comp_octree_)->isNodeOccupied(*it)) {
      Eigen::Vector3d query_point(it.getX(), it.getY(), it.getZ());
      query_point = T_base_comp_ * query_point;
      octomap::OcTreeNode* base_node;  // Keep node to check occupancy.
      if (base_octree_->isObserved(query_point, &base_node)) {
        observed_points->push_back(query_point);
        // If base tree node is occupied distance is very low -> set 0.
        if (params_.k_nearest_neighbor == 1 &&
            base_node &&
            (*base_octree_)->isNodeOccupied(base_node)) {
          distances->push_back(Eigen::Matrix<double, 1, 1>::Zero());
          // TODO: find a better way to get key from node poniter.
          keys->push_back((*base_octree_)->coordToKey(octomap::point3d(query_point.x(),
                                                                       query_point.y(),
                                                                       query_point.z())));
        }
        else {
          OctomapContainer::KNNResult knn_result =
              base_octree_->findKNN(query_point, params_.k_nearest_neighbor);
          distances->push_back(knn_result.distances);
          keys->push_back(knn_result.keys.front());
          if (knn_result.distances(0) > max_dist) max_dist = knn_result.distances(0);
        }
      }
      else {
        unobserved_points->push_back(query_point);
      }
    }
  }
  return max_dist;
}

double OctomapCompare::compareBackward(const KeyToDistMap& key_to_dist,
                                     double max_dist,
                                     std::list<Eigen::Vector3d>* observed_points,
                                     std::list<Eigen::VectorXd>* distances,
                                     std::list<Eigen::Vector3d>* unobserved_points) {
  CHECK_NOTNULL(observed_points);
  CHECK_NOTNULL(distances);
  CHECK_NOTNULL(unobserved_points);
  // Compare base octree to comp octree.
  for (auto it = (*base_octree_)->begin_leafs(); it != (*base_octree_)->end_leafs(); ++it) {
    Eigen::Vector3d query_point(it.getX(), it.getY(), it.getZ());
    if ((*base_octree_)->isNodeOccupied(*it)) {
      Eigen::Vector3d query_point_comp = T_comp_base_ * query_point;
      octomap::OcTreeNode* comp_node;
      if (comp_octree_->isObserved(query_point_comp, &comp_node)) {
        observed_points->push_back(query_point);
        // Check if comp node is also occupied.
        if (params_.k_nearest_neighbor == 1 &&
            comp_node &&
            (*comp_octree_)->isNodeOccupied(comp_node)) {
          distances->push_back(Eigen::Matrix<double, 1, 1>::Zero());
        }
        else {
          // Check if we already know the distance.
          octomap::OcTreeKey base_key = it.getKey();
          auto key_iter = key_to_dist.find(base_key);
          if (key_iter != key_to_dist.end()) {
            distances->push_back(Eigen::Matrix<double, 1, 1>::Constant(key_iter->second));
          }
          else {
            OctomapContainer::KNNResult knn_result =
                comp_octree_->findKNN(query_point_comp, params_.k_nearest_neighbor);
            distances->push_back(knn_result.distances);
            if (knn_result.distances(0) > max_dist) max_dist = knn_result.distances(0);
          }
        }
      }
      else {
        unobserved_points->push_back(query_point);
      }
    }
  }
  return max_dist;
}

void OctomapCompare::getTransformFromICP() {
  PM::ICP icp;
  icp.setDefault();
  PM::DataPoints base_points = matrix3dEigenToPointMatcher(base_octree_->Points());
  PM::DataPoints comp_points = matrix3dEigenToPointMatcher(comp_octree_->Points());

  PM::TransformationParameters T = icp(comp_points, base_points);
  std::cout << "Done ICPeeing, transform is\n"<< T << "\n";
  T_base_comp_.matrix() = T;
  T_comp_base_ = T_base_comp_.inverse();
}

void OctomapCompare::getChanges(const CompareResult& result,
                                Eigen::Matrix<double, 3, Eigen::Dynamic>* output,
                                Eigen::VectorXi* cluster) {
  CHECK_NOTNULL(output);
  // Apply threshold.
  std::list<unsigned int> comp_indices;
  const double thres = params_.distance_threshold * params_.distance_threshold;
  for (unsigned int i = 0; i < result.comp_distances.cols(); ++i) {
    if (result.comp_distances(0,i) > thres) comp_indices.push_back(i);
  }
  std::list<unsigned int> base_indices;
  for (unsigned int i = 0; i < result.base_distances.cols(); ++i) {
    if (result.base_distances(0,i) > thres) base_indices.push_back(i);
  }
  output->resize(3, comp_indices.size() + base_indices.size());
  size_t counter = 0;
  for (auto&& index : comp_indices) {
    output->col(counter++) = result.comp_observed_points.col(index);
  }
  for (auto&& index : base_indices) {
    output->col(counter++) = result.base_observed_points.col(index);
  }
  std::cout << output->cols() << " voxels left after distance thresholding\n";
  // Copy of transpose is necessary because dbscan works with a reference and has issues when
  // directly passing output->transpose().
  Eigen::MatrixXd transpose(output->transpose());
  // DBSCAN filtering.
  Dbscan dbscan(transpose, params_.eps, params_.min_pts);
  dbscan.cluster(cluster);

}

void OctomapCompare::compareResultToPointCloud(const CompareResult& result,
                                               pcl::PointCloud<pcl::PointXYZRGB>* distance_point_cloud) {
  CHECK_NOTNULL(distance_point_cloud)->clear();
  distance_point_cloud->header.frame_id = "map";
  const double max_dist = sqrt(result.max_dist);
  for (unsigned int i = 0; i < result.base_observed_points.cols(); ++i) {
    pcl::RGB color = getColorFromDistance(sqrt(result.base_distances(0,i)), max_dist);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.base_observed_points(0,i);
    pointrgb.y = result.base_observed_points(1,i);
    pointrgb.z = result.base_observed_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
  for (unsigned int i = 0; i < result.comp_observed_points.cols(); ++i) {
    pcl::RGB color = getColorFromDistance(sqrt(result.comp_distances(0,i)), max_dist);
    pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
    pointrgb.x = result.comp_observed_points(0,i);
    pointrgb.y = result.comp_observed_points(1,i);
    pointrgb.z = result.comp_observed_points(2,i);
    distance_point_cloud->push_back(pointrgb);
  }
  if (params_.show_unobserved_voxels) {
    for (unsigned int i = 0; i < result.unobserved_points.cols(); ++i) {
      pcl::RGB color = getColorFromDistance(-1.0);
      pcl::PointXYZRGB pointrgb(color.r, color.g, color.b);
      pointrgb.x = result.unobserved_points(0,i);
      pointrgb.y = result.unobserved_points(1,i);
      pointrgb.z = result.unobserved_points(2,i);
      distance_point_cloud->push_back(pointrgb);
    }
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
  else if (dist >= 0) {
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
