// C++ program for the above approach 
#include "ubersnap.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector> 
using namespace cv; 
using namespace std; 

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_path> <function_name1> [arg1] <function_name2> [arg2] ...\n";
        return 1;
    }

    string imagePath = argv[1];
    try{
        // Load the image
        Mat original_image = imread(imagePath);
        Mat image = imread(imagePath);

        if (image.empty()) {
            cerr << "Error loading image: " << imagePath << "\n";
            return 1;
        }

        // Parse command-line arguments
        vector<string> functionNames;
        vector<double> functionArgs;

        for (int i = 2; i < argc; ++i) {
            if (i + 1 < argc) {
                functionNames.push_back(argv[i]);
                functionArgs.push_back(stod(argv[i + 1]));
                ++i;  // Skip the next argument
            } else {
                cerr << "Invalid number of arguments for function: " << argv[i] << "\n";
                return 1;
            }
        }

        // Run the specified functions
        for (size_t i = 0; i < functionNames.size(); ++i) {
            const string& functionName = functionNames[i];
            double arg = functionArgs[i];

            if (functionName == "brightnes") {
                image = increase_brightness(image, arg);
            } 
            else if (functionName == "highlight") {
                image = adjust_highlight(image, arg);
            } 
            else if (functionName == "whites") {
                image = adjust_whites(image,arg);
            } 
            else if (functionName == "exposure") {
                image = adjust_exposure_cpp(image,arg);
            } 
            else if (functionName == "contrast") {
                image = adjust_contrast(image,arg);
            } 
            else if (functionName == "shadow") {
                image = adjust_shadows(image,arg);
            } 
            else if (functionName == "black") {
                image = adjust_blacks(image,arg);
            } 
            else if (functionName == "saturation") {
                image = adjust_saturation(image,arg);
            } 
            else if (functionName == "sharpnes") {
                image = adjust_sharpness(image,arg);
            } 
            else if (functionName == "clarity") {
                image = adjust_clarity(image,arg);
            } 
            else if (functionName == "temperature") {
                image = adjust_temperature(image,arg);
            } 
            else if (functionName == "tint") {
                image = adjust_tint(image,arg);
            } 
            else {
                cerr << "Function not found: " << functionName << "\n";
                return 1;
            }
        }

        // Display or save the modified image as needed
        cv::imshow("original Image", original_image);
        cv::imshow("Modified Image", image);
        cv::waitKey(0);

        return 0;
    }catch(Exception& e) {
        cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
    }

    
}
