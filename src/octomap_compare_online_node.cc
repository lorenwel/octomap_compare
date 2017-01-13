#include <chrono>

#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pointmatcher_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

#include "octomap_compare/change_map.h"
#include "octomap_compare/file_writer.h"
#include "octomap_compare/load_parameters.h"
#include "octomap_compare/octomap_compare.h"
#include "octomap_compare/octomap_compare_utils.h"

class Online {

  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber transform_sub_;
  ros::Publisher change_candidates_pub_;
  ros::Publisher changes_pub_;
  ros::Publisher heat_map_pub_;
  ros::Publisher change_map_pub_;
  tf::TransformListener tf_listener_;

  OctomapCompare octomap_compare_;
  OctomapCompare::CompareParams params_;

  RandomForestClassifier classifier_;

  Eigen::Affine3d relocalization_transform_;

  ChangeMap<pcl::PointXYZRGB> change_map_;

  size_t n_printed_;

  bool first_relocalization_received_;

  void cloudCallback(const sensor_msgs::PointCloud2& cloud) {
    if (cloud.width * cloud.height == 0) {
      ROS_WARN("Received empty cloud");
      return;
    }
    tf::StampedTransform T_temp;
    if (first_relocalization_received_) {
      try {
        Timer pipeline_timer("pipeline");
        Timer init_timer("init");
        tf_listener_.lookupTransform("map",
                                     cloud.header.frame_id,
                                     cloud.header.stamp,
                                     T_temp);
        Eigen::Affine3d T_map_robot;
        tf::transformTFToEigen(T_temp, T_map_robot);
        Eigen::Affine3d T_initial = relocalization_transform_ * T_map_robot;

        PM::DataPoints intermediate_points =
            PointMatcher_ros::rosMsgToPointMatcherCloud<double>(cloud);
        Matrix3xDynamic points = pointMatcherToMatrix3dEigen(intermediate_points);
        init_timer.stop();

        Timer compare_timer("compare");
        PointCloudContainer compare_container(points, params_.spherical_transform, params_.std_dev);

        octomap_compare_.compare(compare_container, &T_initial.matrix());
        compare_timer.stop();

        pcl::PointCloud<pcl::PointXYZRGB> change_candidate_point_cloud;
        octomap_compare_.getChangeCandidates(&change_candidate_point_cloud);

        Timer cluster_extraction_timer("cluster_extraction");
        std::vector<Cluster> clusters;
        octomap_compare_.getClusters(&clusters);
        cluster_extraction_timer.stop();

        Timer classification_timer("classification");
        std::vector<bool> labels;
        classifier_.classify(clusters, &labels);
        classification_timer.stop();

        // Publish changes. TODO: Make this a function'n'stuff.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr changes_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        changes_point_cloud->header.frame_id = "map";
        pcl::PointCloud<pcl::PointXYZRGB> temp_cloud;
        for (size_t i = 0; i < clusters.size(); ++i) {
          if (labels[i]) {
            clusterToPointCloud(clusters[i], &temp_cloud, T_initial);
            *changes_point_cloud += temp_cloud;
          }
        }
        change_map_.addPointCloud(changes_point_cloud);

        pcl::PointCloud<pcl::PointXYZRGB> heat_map_point_cloud;
        octomap_compare_.getDistanceHeatMap(&heat_map_point_cloud);

        heat_map_pub_.publish(heat_map_point_cloud);
        change_candidates_pub_.publish(change_candidate_point_cloud);
        changes_pub_.publish(changes_point_cloud);
        change_map_pub_.publish(change_map_.getCloud());

        // Correct relocalization transform with ICP transform change.
        relocalization_transform_ = T_initial * T_map_robot.inverse();

        // This part is for labeling data. Stops the pipeline until there's user interaction.
        // BE VERY WARY OF UNCOMMENTING THIS!
//        const std::string filename("/tmp/compare_output_" + std::to_string(n_printed_++) + ".csv");
//        ClusterCentroidVector cluster_centroids;
//        octomap_compare_.saveClusterResultToFile(filename, &cluster_centroids);
//        FileWriter(nh_, cluster_centroids, filename, cloud.header.stamp);

        pipeline_timer.stop();
      }
      catch (tf::TransformException& e) {
        ROS_ERROR("Could not transform point cloud: %s", e.what());
        ROS_WARN_ONCE("This is expected for the first call back");
      }
    }
    else {
      ROS_INFO_ONCE("Waiting to receive first relocalization transform...");
    }
  }

  void transformCallback(const geometry_msgs::Transform& transform) {
    ROS_INFO_ONCE("Received relocalization transform!");
    if (!first_relocalization_received_) {
      first_relocalization_received_ = true;
      tf::transformMsgToEigen(transform, relocalization_transform_);
    }
  }

public:

  Online(const ros::NodeHandle& nh,
         const std::string& base_file,
         const std::string& cloud_topic,
         const OctomapCompare::CompareParams& params) :
         nh_(nh), octomap_compare_(base_file, params), params_(params),
         classifier_(getRandomForestParams(nh_)), change_map_(params.filter_resolution),
         n_printed_(0) {
    cloud_sub_ = nh_.subscribe(cloud_topic, 1, &Online::cloudCallback, this);
    change_candidates_pub_ =
        nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("change_candidates", 1, true);
    changes_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("changes", 1, true);
    heat_map_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("heat_map", 1, true);
    change_map_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("change_map", 1, true);

    relocalization_transform_ = Eigen::Affine3d::Identity();
    bool use_relocalization = false;
    nh_.param("use_relocalization", use_relocalization, use_relocalization);
    if (use_relocalization) {
      first_relocalization_received_ = false;
      transform_sub_ =
          nh_.subscribe("/segmatch/last_transformation", 1, &Online::transformCallback, this);
    }
    else {
      first_relocalization_received_ = true;
    }
  }

  ~Online() {
    sm::timing::Timing::print(std::cout);
  }

};

int main(int argc, char** argv) {
  ros::init(argc, argv, "octomap_compare");
  google::InitGoogleLogging(argv[0]);

  ros::NodeHandle nh("~");

  // Get parameters.
  OctomapCompare::CompareParams params = getCompareParams(nh);
  std::string base_file;
  if (!nh.getParam("base_file", base_file)) {
    LOG(FATAL) << "Did not find base file parameter";
  }
  std::string cloud_topic;
  if (!nh.getParam("cloud_topic", cloud_topic)) {
    ROS_WARN("Did not find cloud_topic parameter. Set to \"/dynamic_point_cloud\"");
  }

  Online online(nh, base_file, cloud_topic, params);

  ros::spin();
}
