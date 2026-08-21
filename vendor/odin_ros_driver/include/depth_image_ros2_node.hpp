/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0.
*/

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "pointcloud_depth_converter.hpp"

class DepthImageRos2Node : public rclcpp::Node
{
public:
    explicit DepthImageRos2Node(
        const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

    void initialize();

private:
    std::string cloud_raw_topic_;
    std::string depth_image_topic_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_;
    std::unique_ptr<PointCloudToDepthConverter> depth_converter_;

    std::chrono::steady_clock::time_point last_receive_time_{};
    std::chrono::steady_clock::time_point last_report_time_{};
    int64_t last_sensor_stamp_ns_ = 0;
    uint64_t received_count_ = 0;
    uint64_t published_count_ = 0;
    uint64_t failed_count_ = 0;
    double max_receive_gap_ms_ = 0.0;
    double max_sensor_gap_ms_ = 0.0;
    double max_processing_ms_ = 0.0;

    PointCloudToDepthConverter::CameraParams loadCameraParams();
    void cloudCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg);
    void recordProcessingTime(
        const std::chrono::steady_clock::time_point &callback_start);
    void reportMetrics(
        const std::chrono::steady_clock::time_point &now);
    void publishDepthImage(const cv::Mat &image,
                           const std_msgs::msg::Header &header);
};
