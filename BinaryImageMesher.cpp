#include "BinaryImageMesher.hpp"

#ifdef REAL
#undef REAL
#endif
#include <opencv2/opencv.hpp>
#define REAL double

#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

namespace triangle {

bool Mesh::save_off(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "OFF" << std::endl;
    out << points.size() << " " << triangles.size() << " 0" << std::endl;

    for (const auto& p : points) {
        out << p.x << " " << p.y << " 0" << std::endl;
    }

    for (const auto& t : triangles) {
        out << "3 " << t.p[0] << " " << t.p[1] << " " << t.p[2] << std::endl;
    }

    return true;
}

BinaryImageMesher::BinaryImageMesher(const BinaryMeshConfig& config)
    : m_config(config) {}

Mesh BinaryImageMesher::triangulate(const cv::Mat& inputImage) {
    m_imgHeight = inputImage.rows;
    cv::Mat gray;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = inputImage.clone();
    }

    cv::Mat binary;
    int thresholdType = (m_config.region == 1) ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV;
    cv::threshold(gray, binary, m_config.thresholdValue, 255, thresholdType);

    std::vector<cv::Point> selectedPixels;
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (binary.at<uchar>(y, x) == 255) {
                selectedPixels.emplace_back(x, y);
            }
        }
    }

    return process_region(selectedPixels);
}

Mesh BinaryImageMesher::process_region(const std::vector<cv::Point>& regionPixels) {
    if (regionPixels.empty()) return Mesh();

    // Find bounding box to create a local mask
    int minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
    for (const auto& p : regionPixels) {
        minX = std::min(minX, p.x); minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x); maxY = std::max(maxY, p.y);
    }

    cv::Mat mask = cv::Mat::zeros(maxY + 1, maxX + 1, CV_8UC1);
    for (const auto& p : regionPixels) {
        mask.at<uchar>(p.y, p.x) = 255;
    }

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    
    Mesh totalMesh;
    totalMesh.success = true;

    for (int i = 1; i < nLabels; ++i) {
        std::vector<cv::Point> componentPixels;
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        for (int y = top; y < top + height; ++y) {
            for (int x = left; x < left + width; ++x) {
                if (labels.at<int>(y, x) == i) {
                    componentPixels.emplace_back(x, y);
                }
            }
        }
        
        Mesh componentMesh = triangulate_component(componentPixels);
        if (componentMesh.success) {
            int offset = (int)totalMesh.points.size();
            for (const auto& p : componentMesh.points) totalMesh.points.push_back(p);
            for (const auto& t : componentMesh.triangles) {
                totalMesh.triangles.emplace_back(t.p[0] + offset, t.p[1] + offset, t.p[2] + offset);
            }
            for (const auto& s : componentMesh.segments) {
                totalMesh.segments.emplace_back(s.p[0] + offset, s.p[1] + offset, s.marker);
            }
        }
    }

    return totalMesh;
}

Mesh BinaryImageMesher::triangulate_component(const std::vector<cv::Point>& componentPixels) {
    if (componentPixels.empty()) return Mesh();

    // 1. Build Initial Mesh
    Mesh mesh = build_initial_mesh(componentPixels);
    if (!mesh.success) return mesh;

    // 2. Topology Analysis
    Topology topo = analyze_topology(mesh);

    // 3. Sequential Smoothing
    apply_boundary_smoothing(mesh, topo);
    apply_interior_smoothing(mesh, topo);

    // 4. Export Final Segments
    for (auto const& e : topo.edgeCounts) {
        if (e.second == 1) {
            mesh.segments.push_back({e.first.first, e.first.second, 0});
        }
    }

    return mesh;
}

Mesh BinaryImageMesher::build_initial_mesh(const std::vector<cv::Point>& pixels) {
    Mesh mesh;
    std::map<std::pair<int, int>, int> globalToLocal;

    auto getOrCreate = [&](int x, int y) -> int {
        auto key = std::make_pair(x, y);
        if (globalToLocal.find(key) == globalToLocal.end()) {
            int localIdx = (int)mesh.points.size();
            globalToLocal[key] = localIdx;
            // Flip Y coordinate to match Cartesian Space (Y-Up)
            mesh.points.emplace_back(x, m_imgHeight - 1 - y, 0, false);
            return localIdx;
        }
        return globalToLocal[key];
    };

    for (const auto& p : pixels) {
        int v0 = getOrCreate(p.x, p.y);
        int v1 = getOrCreate(p.x + 1, p.y);
        int v2 = getOrCreate(p.x + 1, p.y + 1);
        int v3 = getOrCreate(p.x, p.y + 1);
        
        // Use CCW winding for Cartesian Space after Y-flip
        mesh.triangles.emplace_back(v0, v2, v1);
        mesh.triangles.emplace_back(v0, v3, v2);
    }

    mesh.success = !mesh.triangles.empty();
    return mesh;
}

Topology BinaryImageMesher::analyze_topology(const Mesh& mesh) {
    Topology topo;
    topo.adj.resize(mesh.points.size());
    topo.isBoundary.assign(mesh.points.size(), false);
    topo.bAdj.resize(mesh.points.size());

    for (const auto& t : mesh.triangles) {
        for (int i = 0; i < 3; ++i) {
            int u = t.p[i], v = t.p[(i + 1) % 3];
            topo.adj[u].push_back(v);
            topo.adj[v].push_back(u); 
            topo.edgeCounts[{std::min(u, v), std::max(u, v)}]++;
        }
    }

    // Ensure neighbors are unique
    for (auto& neighbors : topo.adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    for (auto const& e : topo.edgeCounts) {
        if (e.second == 1) {
            int u = e.first.first;
            int v = e.first.second;
            topo.isBoundary[u] = topo.isBoundary[v] = true;
            topo.bAdj[u].push_back(v);
            topo.bAdj[v].push_back(u);
        }
    }
    return topo;
}

void BinaryImageMesher::apply_boundary_smoothing(Mesh& mesh, const Topology& topo) {
    const double weight = 0.5;
    for (int iter = 0; iter < m_config.smoothItersBoundary; ++iter) {
        std::vector<Point2D> next = mesh.points;
        for (size_t i = 0; i < mesh.points.size(); ++i) {
            if (topo.isBoundary[i] && !topo.bAdj[i].empty()) {
                double ax = 0, ay = 0;
                for (int nb : topo.bAdj[i]) {
                    ax += mesh.points[nb].x;
                    ay += mesh.points[nb].y;
                }
                ax /= topo.bAdj[i].size();
                ay /= topo.bAdj[i].size();
                next[i].x = mesh.points[i].x + weight * (ax - mesh.points[i].x);
                next[i].y = mesh.points[i].y + weight * (ay - mesh.points[i].y);
            }
        }
        mesh.points = next;
    }
}

void BinaryImageMesher::apply_interior_smoothing(Mesh& mesh, const Topology& topo) {
    const double weight = 0.5;
    for (int iter = 0; iter < m_config.smoothItersInterior; ++iter) {
        std::vector<Point2D> next = mesh.points;
        for (size_t i = 0; i < mesh.points.size(); ++i) {
            if (!topo.isBoundary[i] && !topo.adj[i].empty()) {
                double ax = 0, ay = 0;
                for (int nb : topo.adj[i]) {
                    ax += mesh.points[nb].x;
                    ay += mesh.points[nb].y;
                }
                ax /= topo.adj[i].size();
                ay /= topo.adj[i].size();
                next[i].x = mesh.points[i].x + weight * (ax - mesh.points[i].x);
                next[i].y = mesh.points[i].y + weight * (ay - mesh.points[i].y);
            }
        }
        mesh.points = next;
    }
}

}
