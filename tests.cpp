#include "BinaryImageMesher.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace triangle;

void test_single_pixel() {
    std::cout << "Running test_single_pixel..." << std::endl;
    BinaryMeshConfig config;
    BinaryImageMesher mesher(config);
    mesher.set_image_height(100);

    std::vector<cv::Point> pixels = {{10, 10}};
    Mesh mesh = mesher.build_initial_mesh(pixels);

    assert(mesh.success);
    assert(mesh.points.size() == 4);
    assert(mesh.triangles.size() == 2);

    Topology topo = mesher.analyze_topology(mesh);
    for (bool b : topo.isBoundary) {
        assert(b == true); // In a single pixel, all 4 corners are boundary
    }
    std::cout << "  Passed!" << std::endl;
}

void test_square_2x2() {
    std::cout << "Running test_square_2x2..." << std::endl;
    BinaryMeshConfig config;
    BinaryImageMesher mesher(config);
    mesher.set_image_height(10); // Use height 10

    // 2x2 square of pixels
    std::vector<cv::Point> pixels = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    Mesh mesh = mesher.build_initial_mesh(pixels);

    assert(mesh.success);
    assert(mesh.points.size() == 9); // (0,0) to (2,2)
    assert(mesh.triangles.size() == 8); // 2 per pixel

    Topology topo = mesher.analyze_topology(mesh);
    
    // The center point (1,1) flipped with height 10 becomes (1, 10-1-1) = (1, 8)
    int centerIdx = -1;
    for (size_t i = 0; i < mesh.points.size(); ++i) {
        if (mesh.points[i].x == 1 && mesh.points[i].y == 8) centerIdx = i;
    }
    assert(centerIdx != -1);
    assert(topo.isBoundary[centerIdx] == false);
    
    // Corners and edges should be boundary
    for (size_t i = 0; i < mesh.points.size(); ++i) {
        if (i != centerIdx) assert(topo.isBoundary[i] == true);
    }
    std::cout << "  Passed!" << std::endl;
}

void test_disconnected_components() {
    std::cout << "Running test_disconnected_components..." << std::endl;
    BinaryMeshConfig config;
    BinaryImageMesher mesher(config);

    // Two pixels far apart
    std::vector<cv::Point> pixels = {{0, 0}, {10, 10}};
    Mesh mesh = mesher.process_region(pixels);

    assert(mesh.success);
    assert(mesh.points.size() == 8); // 4 for each pixel
    assert(mesh.triangles.size() == 4); // 2 for each pixel
    std::cout << "  Passed!" << std::endl;
}

void test_smoothing() {
    std::cout << "Running test_smoothing..." << std::endl;
    BinaryMeshConfig config;
    config.smoothItersBoundary = 1;
    config.smoothItersInterior = 1;
    BinaryImageMesher mesher(config);

    // L-shape: (0,0), (1,0), (0,1)
    std::vector<cv::Point> pixels = {{0, 0}, {1, 0}, {0, 1}};
    Mesh mesh = mesher.build_initial_mesh(pixels);
    Point2D originalPoint = mesh.points[0];

    Topology topo = mesher.analyze_topology(mesh);
    mesher.apply_boundary_smoothing(mesh, topo);

    // Check that boundary points actually moved
    bool moved = false;
    for (size_t i = 0; i < mesh.points.size(); ++i) {
        if (mesh.points[i].x != originalPoint.x || mesh.points[i].y != originalPoint.y) {
            moved = true;
            break;
        }
    }
    assert(moved);
    std::cout << "  Passed!" << std::endl;
}

int main() {
    try {
        test_single_pixel();
        test_square_2x2();
        test_disconnected_components();
        test_smoothing();
        std::cout << "\nAll tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
