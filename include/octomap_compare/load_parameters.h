#ifndef OCTOMAP_COMPARE_LOAD_PARAMETERS_H_
#define OCTOMAP_COMPARE_LOAD_PARAMETERS_H_

#include <ros/ros.h>

#include "octomap_compare/feature_extractor.h"
#include "octomap_compare/octomap_compare.h"
#include "octomap_compare/random_forest_classifier.h"

FeatureExtractor::HistogramParams getHistogramParams(ros::NodeHandle& nh) {
  FeatureExtractor::HistogramParams params;

  nh.param("classifier/histogram/min_val", params.min_val, params.min_val);
  int n_bins(params.n_bins);
  nh.param("classifier/histogram/n_bins", n_bins, n_bins);
  params.n_bins = n_bins;
  nh.param("classifier/histogram/bin_size", params.bin_size,
                                 params.bin_size);

  return params;
}

RandomForestClassifier::Params getRandomForestParams(ros::NodeHandle& nh) {
  RandomForestClassifier::Params params;

  params.histogram_params = getHistogramParams(nh);

  nh.param("classifier_file_name", params.classifier_file_name, params.classifier_file_name);
  nh.param("classifier/roc_filename", params.roc_filename, params.roc_filename);
  nh.param("classifier/probability_threshold",
           params.probability_threshold,
           params.probability_threshold);
  nh.param("classifier/save_classifier_after_training",
           params.save_classifier_after_training,
           params.save_classifier_after_training);

  nh.param("classifier/rf_max_depth", params.rf_max_depth, params.rf_max_depth);
  nh.param("classifier/rf_priors", params.rf_priors, params.rf_priors);
  nh.param("classifier/rf_use_surrogates", params.rf_use_surrogates, params.rf_use_surrogates);
  nh.param("classifier/rf_n_active_vars", params.rf_n_active_vars, params.rf_n_active_vars);
  nh.param("classifier/rf_accuracy", params.rf_accuracy, params.rf_accuracy);
  nh.param("classifier/rf_min_sample_ratio",
           params.rf_min_sample_ratio,
           params.rf_min_sample_ratio);
  nh.param("classifier/rf_calc_var_importance",
           params.rf_calc_var_importance,
           params.rf_calc_var_importance);
  nh.param("classifier/rf_max_num_of_trees",
           params.rf_max_num_of_trees,
           params.rf_max_num_of_trees);

  // The following are not actually used by the random tree but need to be set regardless.
  nh.param("classifier/rf_max_categories", params.rf_max_categories, params.rf_max_categories);
  nh.param("classifier/rf_regression_accuracy",
           params.rf_regression_accuracy,
           params.rf_regression_accuracy);

  return params;
}

OctomapCompare::CompareParams getCompareParams(ros::NodeHandle& nh) {
  OctomapCompare::CompareParams params;

  nh.param("max_vis_dist", params.max_vis_dist, params.max_vis_dist);
  nh.param("distance_threshold", params.distance_threshold, params.distance_threshold);
  nh.param("eps", params.eps, params.eps);
  nh.param("min_pts", params.min_pts, params.min_pts);
  nh.param("k_nearest_neighbor", params.k_nearest_neighbor, params.k_nearest_neighbor);
  nh.param("show_unobserved_voxels", params.show_unobserved_voxels, params.show_unobserved_voxels);
  nh.param("show_outliers", params.show_outliers, params.show_outliers);
  nh.param("distance_computation", params.distance_computation, params.distance_computation);
  nh.param("color_changes", params.color_changes, params.color_changes);
  nh.param("perform_icp", params.perform_icp, params.perform_icp);
  nh.param("clustering_algorithm", params.clustering_algorithm, params.clustering_algorithm);
  nh.param("clustering_space", params.clustering_space, params.clustering_space);
  nh.param("min_overlap_ratio", params.min_overlap_ratio, params.min_overlap_ratio);
  int min_num_overlap;
  nh.param("min_num_overlap", min_num_overlap, min_num_overlap);
  params.min_num_overlap = min_num_overlap;
//  nh.getParam("/laser_mapper/icp_configuration_file", params.icp_configuration_file);
//  nh.getParam("/laser_mapper/icp_input_filters_file", params.icp_input_filters_file);
//  nh.getParam("/laser_mapper/icp_input_filters_file", params.icp_base_filters_file);
  std::vector<double> temp_transform({1, 0, 0, 0, 1, 0, 0, 0, 1});
  nh.param("spherical_transform", temp_transform, temp_transform);
  params.spherical_transform = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(temp_transform.data());
  std::vector<double> std_dev_vec;
  if (!nh.getParam("std_dev", std_dev_vec)) {
    LOG(FATAL) << "No standard deviation specified";
  }
  else {
    params.std_dev = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(std_dev_vec.data());
  }

  return params;
}

#endif /* OCTOMAP_COMPARE_LOAD_PARAMETERS_H_ */
