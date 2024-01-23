#include "ubersnap.h"

#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace cv;
using namespace std; 

Mat increase_brightness(const Mat& img, int value) {
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    vector<Mat> channels;
    split(hsv, channels);

    int lim = 255 - value;
    threshold(channels[2], channels[2], lim, 255, THRESH_BINARY);
    channels[2] += value;

    merge(channels, hsv);

    Mat result;
    cvtColor(hsv, result, COLOR_HSV2BGR);

    return result;
}

Mat adjust_highlight(const Mat& image, double factor) {
    factor /= 1000.0;

    // Convert the image to LAB color space
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Split the LAB image into L, A, and B channels
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);

    // Adjust the L channel (lightness) to change highlights
    lab_channels[0] = min(255, max(0, lab_channels[0] * (1 + factor)));

    // Merge the LAB channels back to form the adjusted LAB image
    Mat adjusted_lab_image;
    merge(lab_channels, adjusted_lab_image);

    // Convert the LAB image back to BGR color space
    Mat adjusted_image;
    cvtColor(adjusted_lab_image, adjusted_image, COLOR_Lab2BGR);

    return adjusted_image;
}

Mat adjust_whites(const Mat& image, double factor) {
    // Convert the image to LAB color space
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Split the LAB image into L, A, and B channels
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);

    // Adjust the L channel (lightness) to change highlights
    lab_channels[0] = min(255, max(0, lab_channels[0] * factor));

    // Merge the LAB channels back to form the adjusted LAB image
    Mat adjusted_lab_image;
    merge(lab_channels, adjusted_lab_image);

    // Convert the LAB image back to BGR color space
    Mat adjusted_image;
    cvtColor(adjusted_lab_image, adjusted_image, COLOR_Lab2BGR);

    return adjusted_image;
}

Mat adjust_contrast(const Mat& image, double factor) {
    // Apply contrast adjustment
    double alpha = 1 + factor / 200.0; // Increase this value to increase contrast, decrease for lower contrast
    double beta = 128.0 - alpha * 128.0;  // You can adjust brightness by changing this value
    Mat adjusted_image;
    convertScaleAbs(image, adjusted_image, alpha, beta);

    return adjusted_image;
}

Mat adjust_shadows(const Mat& image, int factor) {
    // Convert the image to LAB color space
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Split the LAB image into L, A, and B channels
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);

    // Apply shadow adjustment to the L channel
    lab_channels[0] = min(255, max(0, lab_channels[0] + factor));

    // Merge the LAB channels back to form the adjusted LAB image
    Mat adjusted_lab_image;
    merge(lab_channels, adjusted_lab_image);

    // Convert the LAB image back to BGR color space
    Mat adjusted_image;
    cvtColor(adjusted_lab_image, adjusted_image, COLOR_Lab2BGR);

    return adjusted_image;
}

Mat adjust_blacks(const Mat& image, double factor) {
    factor /= 200.0;

    // Convert the image to LAB color space
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Split the LAB image into L, A, and B channels
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);

    // Calculate a threshold to identify the darker areas
    double threshold = mean(lab_channels[0])[0];

    // Adjust the darker areas based on the factor
    for (int i = 0; i < lab_channels[0].rows; ++i) {
        for (int j = 0; j < lab_channels[0].cols; ++j) {
            if (factor > 0.0) {  // Darken
                if (lab_channels[0].at<uchar>(i, j) < threshold) {
                    lab_channels[0].at<uchar>(i, j) = saturate_cast<uchar>(max(0, static_cast<int>(lab_channels[0].at<uchar>(i, j) - ((threshold - lab_channels[0].at<uchar>(i, j)) * factor))));
                }
            } else if (factor < 0.0) {  // Lighten
                if (lab_channels[0].at<uchar>(i, j) < threshold) {
                    lab_channels[0].at<uchar>(i, j) = saturate_cast<uchar>(min(255, static_cast<int>(lab_channels[0].at<uchar>(i, j) - ((threshold - lab_channels[0].at<uchar>(i, j)) * factor))));
                }
            }
        }
    }

    // Merge the LAB channels back to form the adjusted LAB image
    Mat adjusted_lab_image;
    merge(lab_channels, adjusted_lab_image);

    // Convert the LAB image back to BGR color space
    Mat adjusted_image;
    cvtColor(adjusted_lab_image, adjusted_image, COLOR_Lab2BGR);

    return adjusted_image;
}

Mat adjust_saturation(const Mat& image, double factor) {
    factor /= 200.0;

    // Convert the image to the HSV color space
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_BGR2HSV);

    // Loop through each pixel and adjust the saturation channel
    for (int i = 0; i < hsv_image.rows; ++i) {
        for (int j = 0; j < hsv_image.cols; ++j) {
            hsv_image.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(max(0, min(255, static_cast<int>(hsv_image.at<Vec3b>(i, j)[1] * (1 + factor)))));
        }
    }

    // Convert the adjusted HSV image back to BGR color space
    Mat adjusted_image;
    cvtColor(hsv_image, adjusted_image, COLOR_HSV2BGR);

    return adjusted_image;
}

Mat adjust_highlights(const Mat& image, double factor) {
    // Convert the image to LAB color space
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Split the LAB image into L, A, and B channels
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);

    // Adjust the L channel (lightness) for highlights
    lab_channels[0] = lab_channels[0] * (1 + factor);

    // Merge the LAB channels back to form the adjusted LAB image
    Mat adjusted_lab_image;
    merge(lab_channels, adjusted_lab_image);


    // Convert the LAB image back to BGR color space
    Mat adjusted_image;
    cvtColor(adjusted_lab_image, adjusted_image, COLOR_Lab2BGR);

    // Clip the pixel values to the valid range [0, 255]
    adjusted_image = min(max(adjusted_image, 0), 255);

    return adjusted_image;
}

Mat adjust_exposure_cpp(const Mat& image, double factor) {
    if (factor > 0) {
        factor /= 150.0;
    }
    if (factor < 0) {
        factor /= 20.0;
    }

    vector<double> gamma_table;
    for (int i = 0; i < 256; ++i) {
        gamma_table.push_back(pow(static_cast<double>(i) / 255.0, (factor * -1.1) + 1.0) * 255.0);
    }

    Mat gamma_mat(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        gamma_mat.at<uchar>(i) = static_cast<uchar>(round(gamma_table[i]));
    }

    Mat adjusted_image;
    LUT(image, gamma_mat, adjusted_image);

    return adjusted_image;
}

Mat adjust_sharpness(const Mat& image, double sharpness_level) {
    sharpness_level += 1.0;
    int kernel_size = 5;  // Adjust for desired sharpening intensity

    // Apply Gaussian blur
    Mat blurred;
    GaussianBlur(image, blurred, Size(kernel_size, kernel_size), 0);

    // Apply sharpening using addWeighted
    Mat sharpened;
    addWeighted(image, 1.5, blurred, -0.5, 0, sharpened);

    // Final sharpening adjustment
    addWeighted(image, 1 - sharpness_level, sharpened, sharpness_level, 0, sharpened);

    return sharpened;
}

Mat adjust_clarity(const Mat& image, double clarity_level) {
    Mat lab;
    cvtColor(image, lab, COLOR_BGR2Lab);

    vector<Mat> lab_channels;
    split(lab, lab_channels);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(lab_channels[0], lab_channels[0]);

    merge(lab_channels, lab);

    Mat bgr;
    cvtColor(lab, bgr, COLOR_BGR2Lab);

    Mat clarity_result;
    addWeighted(image, 1 - clarity_level, bgr, clarity_level, 0, clarity_result);

    return clarity_result;
}

Mat adjust_temperature(Mat& image, double temperature_level) {
    temperature_level /= 2.0;
    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Access blue and red channels directly using indexing
            int b_value = image.at<Vec3b>(y, x)[0];
            int r_value = image.at<Vec3b>(y, x)[2];

            // Shift blue and red channels in opposite directions
            image.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(min(max(b_value - temperature_level, 0.0), 255.0));
            image.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(min(max(r_value + temperature_level, 0.0), 255.0));
        }
    }

    return image;
}

Mat adjust_tint(Mat& image, double tint_level) {
    tint_level /= 2.0;
    int height = image.rows;
    int width = image.cols;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Access blue and red channels directly using indexing
            int b_value = image.at<Vec3b>(y, x)[0];
            int r_value = image.at<Vec3b>(y, x)[2];

            // Shift blue and red channels in opposite directions
            image.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(min(max(b_value + tint_level, 0.0), 255.0));
            image.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(min(max(r_value + tint_level, 0.0), 255.0));
        }
    }

    return image;
}

Mat adjust_exposure(const Mat& image, double factor) {
    Mat adjusted_image;
    double alpha = 0.5;
    double beta = 205.0;

    convertScaleAbs(image, adjusted_image, alpha, beta);

    return adjusted_image;
}

Mat adjust_exposure_manual(Mat& image, double factor) {
    if (factor > 0) {
        factor /= 110;
    }
    if (factor < 0) {
        factor /= 20;
    }

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                int pixel = image.at<Vec3b>(y, x)[c];
                int adjusted_pixel = min(255, static_cast<int>(255 * pow(pixel / 255.0, (factor * -1) + 1)));
                image.at<Vec3b>(y, x)[c] = adjusted_pixel;
            }
        }
    }

    return image;
}

Mat adjust_brightness_alpha(const Mat& image, double factor) {
    // Apply brightness adjustment
    double alpha = 1 + factor / 200.0;
    double beta = 1.0;
    Mat adjusted_image;

    convertScaleAbs(image, adjusted_image, alpha, beta);

    return adjusted_image;
}

Mat adjust_exposure_weighted(const Mat& img, double value) {
    Mat adjusted_img;
    addWeighted(img, 1.0, Mat::zeros(img.size(), img.type()), 1.0, value, adjusted_img);

    return adjusted_img;
}

Mat adjust_brightness_with_beta(const Mat& img, double value) {
    // Apply brightness adjustment using beta value
    double alpha = 1.0;
    double beta = value;

    Mat result_img;
    convertScaleAbs(img, result_img, alpha, beta);

    return result_img;
}
