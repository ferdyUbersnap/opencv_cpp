#ifndef UBERSNAP_H
#define UBERSNAP_H
#include <opencv2/opencv.hpp>
using namespace cv; 

// Function declaration
Mat increase_brightness(const Mat& img, int value); 
Mat adjust_highlight(const Mat& image, double factor);
Mat adjust_whites(const Mat& image, double factor);
Mat adjust_contrast(const Mat& image, double factor);
Mat adjust_shadows(const Mat& image, int factor);
Mat adjust_blacks(const Mat& image, double factor);
Mat adjust_saturation(const Mat& image, double factor);
Mat adjust_highlights(const Mat& image, double factor);
Mat adjust_exposure_cpp(const Mat& image, double factor);
Mat adjust_sharpness(const Mat& image, double sharpness_level);
Mat adjust_clarity(const Mat& image, double clarity_level);
Mat adjust_temperature(Mat& image, double temperature_level);
Mat adjust_tint(Mat& image, double tint_level);
Mat adjust_exposure(const Mat& image, double factor);
Mat adjust_exposure_manual(Mat& image, double factor);
Mat adjust_brightness_alpha(const Mat& image, double factor);
Mat adjust_exposure_weighted(const Mat& img, double value);
Mat adjust_brightness_with_beta(const Mat& img, double value);


#endif