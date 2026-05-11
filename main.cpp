#include "BinaryImageMesher.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

void print_help(const char* progName) {
    std::cout << "BinaryImageMesher CLI" << std::endl;
    std::cout << "Usage: " << progName << " <image_path> [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -h, --help            Show this help message" << std::endl;
    std::cout << "  -t, --threshold VAL   Threshold value (0-255, default: 127)" << std::endl;
    std::cout << "  -r, --region 0|1      Region to mesh (1: White/Foreground, 0: Black/Background, default: 1)" << std::endl;
    std::cout << "  -si, --smooth-int N   Number of interior smoothing iterations (default: 10)" << std::endl;
    std::cout << "  -sb, --smooth-bnd N   Number of boundary smoothing iterations (default: 5)" << std::endl;
    std::cout << "  -o, --output FILE     Output OFF filename (default: output.off)" << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << progName << " input.png -t 150 -si 20 -sb 10 -o my_mesh.off" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 0;
    }

    std::string imagePath;
    std::string offPath = "output.off";
    triangle::BinaryMeshConfig config;
    config.thresholdValue = 127;
    config.smoothItersInterior = 10;
    config.smoothItersBoundary = 5;
    config.region = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            config.thresholdValue = std::stoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--region") && i + 1 < argc) {
            config.region = std::stoi(argv[++i]);
        } else if ((arg == "-si" || arg == "--smooth-int") && i + 1 < argc) {
            config.smoothItersInterior = std::stoi(argv[++i]);
        } else if ((arg == "-sb" || arg == "--smooth-bnd") && i + 1 < argc) {
            config.smoothItersBoundary = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            offPath = argv[++i];
        } else if (imagePath.empty()) {
            imagePath = arg;
        }
    }

    if (imagePath.empty()) {
        std::cerr << "Error: No input image specified." << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Error: Could not read image: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Processing " << imagePath << "..." << std::endl;
    std::cout << "Settings: Threshold=" << config.thresholdValue 
              << ", Region=" << (config.region == 1 ? "White" : "Black")
              << ", SmoothInt=" << config.smoothItersInterior 
              << ", SmoothBnd=" << config.smoothItersBoundary << std::endl;

    triangle::BinaryImageMesher mesher(config);
    triangle::Mesh mesh = mesher.triangulate(img);

    if (mesh.success) {
        std::cout << "Mesh generated successfully!" << std::endl;
        std::cout << "Points: " << mesh.points.size() << std::endl;
        std::cout << "Triangles: " << mesh.triangles.size() << std::endl;

        // Visualize result
        cv::Mat display = img.clone();
        for (const auto& t : mesh.triangles) {
            cv::Point p1(mesh.points[t.p[0]].x, mesh.points[t.p[0]].y);
            cv::Point p2(mesh.points[t.p[1]].x, mesh.points[t.p[1]].y);
            cv::Point p3(mesh.points[t.p[2]].x, mesh.points[t.p[2]].y);
            cv::line(display, p1, p2, cv::Scalar(0, 255, 0), 1);
            cv::line(display, p2, p3, cv::Scalar(0, 255, 0), 1);
            cv::line(display, p3, p1, cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite("mesh_output.png", display);
        std::cout << "Saved visualization to mesh_output.png" << std::endl;

        if (mesh.save_off(offPath)) {
            std::cout << "Saved mesh to " << offPath << std::endl;
        } else {
            std::cerr << "Error: Failed to save " << offPath << std::endl;
        }
    } else {
        std::cerr << "Error: Failed to generate mesh." << std::endl;
    }

    return 0;
}
