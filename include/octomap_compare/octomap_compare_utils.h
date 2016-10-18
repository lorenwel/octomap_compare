#ifndef OCTOMAP_COMPARE_UTILS_H_
#define OCTOMAP_COMPARE_UTILS_H_

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pointmatcher/PointMatcher.h>

#include "octomap_compare/octomap_container.h"

typedef PointMatcher<double> PM;

/// \brief Creates new OctomapContainer object with Octomap created from arg.
template <typename T>
void loadOctomap(const T& arg, std::shared_ptr<OctomapContainer>* container) {
  *container = std::make_shared<OctomapContainer>(arg);
}

/// \brief Converts a 3xN Eigen matrix to a PointMatcher cloud.
PM::DataPoints matrix3dEigenToPointMatcher(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& matrix) {
  PM::DataPoints::Labels xyz1;
  xyz1.push_back(PM::DataPoints::Label("x", 1));
  xyz1.push_back(PM::DataPoints::Label("y", 1));
  xyz1.push_back(PM::DataPoints::Label("z", 1));
  xyz1.push_back(PM::DataPoints::Label("pad", 1));
  PM::DataPoints point_matcher(xyz1, PM::DataPoints::Labels(), matrix.cols());
  point_matcher.getFeatureViewByName("x") = matrix.row(0);
  point_matcher.getFeatureViewByName("y") = matrix.row(1);
  point_matcher.getFeatureViewByName("z") = matrix.row(2);
  point_matcher.getFeatureViewByName("pad").setConstant(1);

  return point_matcher;
}

#endif /* OCTOMAP_COMPARE_UTILS_H_ */
