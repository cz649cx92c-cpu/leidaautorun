/*
Copyright 2025 Manifold Tech Ltd.(www.manifoldtech.com.co)
Licensed under the Apache License, Version 2.0.
*/

#include "pointcloud_depth_converter.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp>
#include <pcl/common/transforms.h>

namespace
{
constexpr int kOutputWidth = 64;
constexpr int kOutputHeight = 40;

inline void keepNearest(cv::Mat &depth, int row, int col, float value)
{
    float &pixel = depth.at<float>(row, col);
    if (pixel == 0.0F || value < pixel) {
        pixel = value;
    }
}
}  // namespace

PointCloudToDepthConverter::PointCloudToDepthConverter(
    const CameraParams &params)
    : params_(params)
{
    initializeInternalParams();
    createDistortionMaps();
}

void PointCloudToDepthConverter::initializeInternalParams()
{
    // Keep the vendor's inexpensive intermediate projection plane (normally
    // about 228x185), but never expand it to the 1600x1296 camera image.
    scaled_width_ = std::max(
        1, static_cast<int>(params_.image_width / params_.scale));
    scaled_height_ = std::max(
        1, static_cast<int>(params_.image_height / params_.scale));

    K_ = Eigen::Matrix3d::Identity();
    K_(0, 0) = params_.A11;
    K_(0, 1) = params_.A12;
    K_(0, 2) = params_.u0;
    K_(1, 1) = params_.A22;
    K_(1, 2) = params_.v0;

    Kl_ = Eigen::Matrix3d::Identity();
    Kl_(0, 0) = params_.A11 / params_.scale;
    Kl_(0, 1) = 0.0;
    Kl_(0, 2) = params_.u0 / params_.scale;
    Kl_(1, 1) = params_.A22 / params_.scale;
    Kl_(1, 2) = params_.v0 / params_.scale;

    K_4x4_ = Eigen::Matrix4d::Identity();
    K_4x4_.block<3, 3>(0, 0) = Kl_;
    Kcl_ = K_4x4_ * params_.Tcl;
}

void PointCloudToDepthConverter::createDistortionMaps()
{
    // The original depth path did not remap depth. These full-resolution maps
    // were used only by the now-disabled RGB colored-cloud demonstration.
    map_x_.release();
    map_y_.release();
    inv_map_x_.release();
    inv_map_y_.release();
}

PointCloudToDepthConverter::ProcessResult
PointCloudToDepthConverter::processCloudAndImage(
    const pcl::PointCloud<pcl::PointXYZ> &cloud, const cv::Mat &image)
{
    ProcessResult result;
    result.success = false;
    const auto validation = validateInputs(cloud, image);
    if (!validation.first) {
        result.error_message = validation.second;
        return result;
    }

    try {
        pcl::PointCloud<pcl::PointXYZ> projected_cloud;
        pcl::transformPointCloud(cloud, projected_cloud, Kcl_);
        result.depth_image = postProcessDepthImage(
            projectCloudToDepth(projected_cloud));
        result.colored_cloud.clear();
        result.success = !result.depth_image.empty();
        if (!result.success) {
            result.error_message = "Depth projection returned an empty image";
        }
    } catch (const std::exception &error) {
        result.error_message = std::string("Processing error: ") + error.what();
    }
    return result;
}

cv::Mat PointCloudToDepthConverter::projectCloudToDepth(
    const pcl::PointCloud<pcl::PointXYZ> &projected_cloud)
{
    cv::Mat depth =
        cv::Mat::zeros(scaled_height_, scaled_width_, CV_32FC1);
    for (const auto &point : projected_cloud) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z) || point.z <= 0.0F) {
            continue;
        }

        const int col = static_cast<int>(std::lround(point.x / point.z));
        const int row = static_cast<int>(std::lround(point.y / point.z));
        if (col < 0 || col >= scaled_width_ ||
            row < 0 || row >= scaled_height_) {
            continue;
        }

        // Retain the original 3x3 densification, but use a z-buffer so a
        // farther point can never overwrite a nearer obstacle.
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                const int target_row = row + dr;
                const int target_col = col + dc;
                if (target_row >= 0 && target_row < scaled_height_ &&
                    target_col >= 0 && target_col < scaled_width_) {
                    keepNearest(depth, target_row, target_col, point.z);
                }
            }
        }
    }
    return depth;
}

cv::Mat PointCloudToDepthConverter::postProcessDepthImage(
    const cv::Mat &depth_img)
{
    if (depth_img.empty() || depth_img.type() != CV_32FC1) {
        return {};
    }

    // Preserve the vendor edge-invalidating behavior on the small projection
    // plane. At the default scale this processes ~42k pixels instead of ~2M.
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Mat magnitude;
    cv::Sobel(depth_img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(depth_img, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, magnitude);

    cv::Mat edge_mask;
    cv::threshold(magnitude, edge_mask, 0.75, 255.0, cv::THRESH_BINARY);
    edge_mask.convertTo(edge_mask, CV_8U);

    cv::Mat filtered = depth_img.clone();
    filtered.setTo(0.0F, edge_mask);

    cv::Mat policy_depth;
    cv::resize(filtered, policy_depth,
               cv::Size(kOutputWidth, kOutputHeight),
               0.0, 0.0, cv::INTER_LINEAR);
    return policy_depth;
}

cv::Mat PointCloudToDepthConverter::customResize(
    const cv::Mat &src, const cv::Size &size)
{
    cv::Mat destination;
    cv::resize(src, destination, size, 0.0, 0.0, cv::INTER_NEAREST);
    return destination;
}

pcl::PointCloud<pcl::PointXYZRGB>
PointCloudToDepthConverter::generateColoredCloud(
    const cv::Mat &, const cv::Mat &)
{
    return {};
}

std::pair<bool, std::string> PointCloudToDepthConverter::validateInputs(
    const pcl::PointCloud<pcl::PointXYZ> &cloud, const cv::Mat &)
{
    if (cloud.empty()) {
        return {false, "Empty point cloud"};
    }
    if (params_.image_width <= 0 || params_.image_height <= 0 ||
        params_.scale <= 0.0 || params_.A11 < 1e-6 || params_.A22 < 1e-6) {
        return {false, "Invalid camera dimensions/intrinsics"};
    }
    return {true, ""};
}

void PointCloudToDepthConverter::updateCameraParams(
    const CameraParams &params)
{
    params_ = params;
    initializeInternalParams();
    createDistortionMaps();
}
