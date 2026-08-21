/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0.
*/

#include "depth_image_ros2_node.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>

DepthImageRos2Node::DepthImageRos2Node(const rclcpp::NodeOptions &options)
    : Node("depth_image_ros2_node", options)
{
    depth_converter_ =
        std::make_unique<PointCloudToDepthConverter>(loadCameraParams());

    cloud_raw_topic_ = this->declare_parameter<std::string>(
        "cloud_raw_topic", "/odin1/cloud_raw");
    depth_image_topic_ = this->declare_parameter<std::string>(
        "depth_image_topic", "/odin1/depth_img_competetion");

    RCLCPP_INFO(this->get_logger(),
                "Low-compute depth: %s -> %s (64x40, 32FC1)",
                cloud_raw_topic_.c_str(), depth_image_topic_.c_str());
}

void DepthImageRos2Node::initialize()
{
    depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        depth_image_topic_, rclcpp::SensorDataQoS().keep_last(1));

    // cloud_raw is a large, fragmented PointCloud2 stream whose vendor
    // publisher is RELIABLE. BEST_EFFORT can discard a whole cloud when one
    // fragment is lost, producing long depth gaps. Keep only the newest frame
    // while retaining reliable delivery.
    const auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                               .reliable()
                               .durability_volatile();
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_raw_topic_, cloud_qos,
        std::bind(&DepthImageRos2Node::cloudCallback, this,
                  std::placeholders::_1));

    last_report_time_ = std::chrono::steady_clock::now();

    RCLCPP_INFO(this->get_logger(),
                "Low-compute SRU depth node initialized (cloud QoS: "
                "reliable, keep_last=1)");
}

void DepthImageRos2Node::cloudCallback(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud_msg)
{
    const auto callback_start = std::chrono::steady_clock::now();
    ++received_count_;

    if (last_receive_time_.time_since_epoch().count() != 0) {
        const double gap_ms = std::chrono::duration<double, std::milli>(
                                  callback_start - last_receive_time_)
                                  .count();
        max_receive_gap_ms_ = std::max(max_receive_gap_ms_, gap_ms);
        if (gap_ms > 300.0) {
            RCLCPP_WARN(this->get_logger(),
                        "cloud_raw receive gap %.1f ms", gap_ms);
        }
    }
    last_receive_time_ = callback_start;

    const int64_t sensor_stamp_ns =
        static_cast<int64_t>(cloud_msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(cloud_msg->header.stamp.nanosec);
    if (last_sensor_stamp_ns_ != 0 && sensor_stamp_ns > last_sensor_stamp_ns_) {
        const double sensor_gap_ms =
            static_cast<double>(sensor_stamp_ns - last_sensor_stamp_ns_) /
            1000000.0;
        max_sensor_gap_ms_ = std::max(max_sensor_gap_ms_, sensor_gap_ms);
    }
    last_sensor_stamp_ns_ = sensor_stamp_ns;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*cloud_msg, cloud);
    if (cloud.empty()) {
        ++failed_count_;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Empty point cloud received");
        recordProcessingTime(callback_start);
        reportMetrics(std::chrono::steady_clock::now());
        return;
    }

    const auto result =
        depth_converter_->processCloudAndImage(cloud, cv::Mat());
    if (!result.success) {
        ++failed_count_;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Depth conversion failed: %s",
                             result.error_message.c_str());
        recordProcessingTime(callback_start);
        reportMetrics(std::chrono::steady_clock::now());
        return;
    }

    publishDepthImage(result.depth_image, cloud_msg->header);
    ++published_count_;
    recordProcessingTime(callback_start);
    reportMetrics(std::chrono::steady_clock::now());
}

void DepthImageRos2Node::recordProcessingTime(
    const std::chrono::steady_clock::time_point &callback_start)
{
    const double processing_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() -
                                     callback_start)
                                     .count();
    max_processing_ms_ = std::max(max_processing_ms_, processing_ms);
}

void DepthImageRos2Node::reportMetrics(
    const std::chrono::steady_clock::time_point &now)
{
    const double elapsed_s =
        std::chrono::duration<double>(now - last_report_time_).count();
    if (elapsed_s < 5.0) {
        return;
    }

    RCLCPP_INFO(
        this->get_logger(),
        "Depth metrics: rx=%.2f Hz, published=%llu, failed=%llu, "
        "max_rx_gap=%.1f ms, max_sensor_gap=%.1f ms, max_processing=%.1f ms",
        static_cast<double>(received_count_) / elapsed_s,
        static_cast<unsigned long long>(published_count_),
        static_cast<unsigned long long>(failed_count_), max_receive_gap_ms_,
        max_sensor_gap_ms_, max_processing_ms_);

    last_report_time_ = now;
    received_count_ = 0;
    published_count_ = 0;
    failed_count_ = 0;
    max_receive_gap_ms_ = 0.0;
    max_sensor_gap_ms_ = 0.0;
    max_processing_ms_ = 0.0;
}

PointCloudToDepthConverter::CameraParams DepthImageRos2Node::loadCameraParams()
{
    PointCloudToDepthConverter::CameraParams params;
    params.image_width = this->declare_parameter<int>("cam_0.image_width", 1600);
    params.image_height = this->declare_parameter<int>("cam_0.image_height", 1296);
    params.A11 = this->declare_parameter<double>("cam_0.A11", 0.0);
    params.A12 = this->declare_parameter<double>("cam_0.A12", 0.0);
    params.A22 = this->declare_parameter<double>("cam_0.A22", 0.0);
    params.u0 = this->declare_parameter<double>("cam_0.u0", 0.0);
    params.v0 = this->declare_parameter<double>("cam_0.v0", 0.0);
    params.k2 = this->declare_parameter<double>("cam_0.k2", 0.0);
    params.k3 = this->declare_parameter<double>("cam_0.k3", 0.0);
    params.k4 = this->declare_parameter<double>("cam_0.k4", 0.0);
    params.k5 = this->declare_parameter<double>("cam_0.k5", 0.0);
    params.k6 = this->declare_parameter<double>("cam_0.k6", 0.0);
    params.k7 = this->declare_parameter<double>("cam_0.k7", 0.0);
    params.scale = this->declare_parameter<double>("scale", 7.0);
    params.point_sampling_rate =
        this->declare_parameter<int>("point_sampling_rate", 5);

    const auto Tcl = this->declare_parameter<std::vector<double>>(
        "Tcl_0", std::vector<double>(16, 0.0));
    if (Tcl.size() != 16) {
        throw std::runtime_error("Tcl_0 must contain 16 values");
    }
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            params.Tcl(row, col) = Tcl[row * 4 + col];
        }
    }

    if (params.image_width <= 0 || params.image_height <= 0 ||
        params.scale <= 0.0 ||
        params.A11 < 1e-6 || params.A22 < 1e-6) {
        throw std::runtime_error("Invalid Odin camera dimensions/intrinsics");
    }
    return params;
}

void DepthImageRos2Node::publishDepthImage(
    const cv::Mat &image, const std_msgs::msg::Header &header)
{
    if (image.empty() || image.type() != CV_32FC1 ||
        !image.isContinuous()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid low-resolution depth image");
        return;
    }

    sensor_msgs::msg::Image message;
    message.header = header;
    message.height = static_cast<uint32_t>(image.rows);
    message.width = static_cast<uint32_t>(image.cols);
    message.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    message.is_bigendian = false;
    message.step = static_cast<sensor_msgs::msg::Image::_step_type>(
        image.cols * sizeof(float));
    message.data.resize(static_cast<size_t>(message.step) * message.height);
    std::memcpy(message.data.data(), image.ptr<float>(), message.data.size());
    depth_image_pub_->publish(message);
}
