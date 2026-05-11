#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/core.hpp>

namespace cv { class Mat; }

namespace triangle {

struct Point2D {
    double x, y;
    int marker;
    bool isFixed;
    Point2D(double _x = 0, double _y = 0, int _m = 0, bool _f = false)
        : x(_x), y(_y), marker(_m), isFixed(_f) {}
};

struct Triangle {
    int p[3];
    Triangle(int p1 = 0, int p2 = 0, int p3 = 0) { p[0] = p1; p[1] = p2; p[2] = p3; }
};

struct Edge {
    int p[2];
    int marker;
    Edge(int p1 = 0, int p2 = 0, int _m = 0) : marker(_m) { p[0] = p1; p[1] = p2; }
};

struct Mesh {
    std::vector<Point2D>  points;
    std::vector<Triangle> triangles;
    std::vector<Edge>     segments;
    bool success = false;
    operator bool() const { return success; }

    bool save_off(const std::string& filename) const;
};

struct Topology {
    std::vector<std::vector<int>> adj;
    std::vector<std::vector<int>> bAdj;
    std::vector<bool> isBoundary;
    std::map<std::pair<int, int>, int> edgeCounts;
};

struct BinaryMeshConfig {
    int region = 1;
    int thresholdValue = 127;
    double minAngle = 20.0;
    int smoothItersInterior = 5;
    int smoothItersBoundary = 2;
};

class BinaryImageMesher {
public:
    BinaryImageMesher(const BinaryMeshConfig& config);
    Mesh triangulate(const cv::Mat& inputImage);

    // Modular methods for testing
    Mesh process_region(const std::vector<cv::Point>& regionPixels);
    Mesh triangulate_component(const std::vector<cv::Point>& componentPixels);
    
    Mesh build_initial_mesh(const std::vector<cv::Point>& pixels);
    Topology analyze_topology(const Mesh& mesh);
    void apply_boundary_smoothing(Mesh& mesh, const Topology& topo);
    void apply_interior_smoothing(Mesh& mesh, const Topology& topo);

    void set_image_height(int h) { m_imgHeight = h; }

private:
    BinaryMeshConfig m_config;
    int m_imgHeight = 0;
};

} // namespace triangle