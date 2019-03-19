#include <iostream>
#include <string>
#include <numeric>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>

void onMouseClick(int event, int x, int y, int flags, void* params);
double gauss(double x, double t_m, double t_sd, double im_m, double im_sd, double p_tumor);
double gauss_t(double x, double t_m, double t_sd);
double gauss_im(double x,  double im_m, double im_sd);
cv::Mat compute_prob(const cv::Mat input, double t_m, double t_sd, double im_m, double im_sd);

struct MouseParams {
    cv::Mat* im;
    std::vector<cv::Point>* coordinates;
    std::vector<double>* intensities;
};

int main() {
    // Load image data
    cv::Mat image;
    image = cv::imread("../data/tumordata.png", cv::IMREAD_GRAYSCALE);
    std::string title {"Data"};


    // Initialize mouse click variables to safe both position and intensity
    std::vector<cv::Point> tumor_points; 
    std::vector<double> intensities;
    MouseParams mp;
    mp.im = &image;
    mp.coordinates = &tumor_points;
    mp.intensities = &intensities;
    
    // Display image and let the user select points from class 1
    int n_points = 20;
    cv::namedWindow(title, 1);
    cv::setMouseCallback(title, onMouseClick, (void*)&mp);
    cv::imshow(title, image);
    
    // Close image after n points are selected
    while(intensities.size() < n_points) {
        cv::waitKey(1);
    }
    cv::destroyAllWindows();

    // Compute mean and standard deviation
    cv::Scalar mean_tumor_tmp, sd_tumor_tmp, mean_image_tmp, sd_image_tmp;
    cv::meanStdDev(image, mean_image_tmp, sd_image_tmp);
    cv::meanStdDev(intensities, mean_tumor_tmp, sd_tumor_tmp);

    double mean_tumor = mean_tumor_tmp[0];
    double mean_image = mean_image_tmp[0];
    double sd_tumor = std::sqrt((sd_tumor_tmp[0] * sd_tumor_tmp[0] * intensities.size()) / (intensities.size() - 1));
    double sd_image = std::sqrt((sd_image_tmp[0] * sd_image_tmp[0] * image.total()) / (image.total() - 1));

    // Compute probability of being a tumor for all pixels
    cv::Mat prob_im = compute_prob(image, mean_tumor, sd_tumor, mean_image, sd_image);
    cv::Mat prob_colormap;
    cv::applyColorMap(prob_im, prob_colormap, 2);
    cv::namedWindow("Probability", 1);
    cv::imshow("Probability", prob_colormap);
    cv::waitKey(0);
    //cv::imwrite("test.png", prob_im);


}

void onMouseClick(int event, int x, int y, int flags, void* params) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        MouseParams* mp = (MouseParams*)params;
        (*(mp->coordinates)).push_back(cv::Point(x,y));
        double intensity = double ( (*(mp->im)).at<uchar>(y,x) );
        (*(mp->intensities)).push_back(intensity);
    }
}

double gauss_im(double x,  double im_m, double im_sd) {
    return 1 / std::sqrt((2*M_PI*im_sd*im_sd)) * std::exp(-( (x-im_m)*(x-im_m) ) / (2*im_sd*im_sd)  );
}

double gauss_t(double x, double t_m, double t_sd) {
    return 1 / std::sqrt((2*M_PI*t_sd*t_sd)) * std::exp(-( (x-t_m)*(x-t_m) ) / (2*t_sd*t_sd)  );   
}

double gauss(double x, double t_m, double t_sd, double im_m, double im_sd, double p_tumor=0.2) {
    // Return probability of pixel being a tumor with given prior
    double p_not = 1 - p_tumor;
    double prob =  (gauss_t(x, t_m, t_sd)*p_tumor) / (gauss_t(x, t_m, t_sd)*p_tumor + gauss_im(x, im_m, im_sd)*p_not); 
    //std::cout << prob << std::endl;
    return prob;
}

cv::Mat compute_prob(const cv::Mat input, double t_m, double t_sd, double im_m, double im_sd) {
    cv::Mat output = input.clone(); 
    
    int nRows = output.rows;
    int nCols = output.cols;

    if (output.isContinuous()) {
        std::cout << "Contionous" << std::endl;
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i) {
        p = output.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j) {           
            p[j] = (uchar)(gauss(double(p[j]), t_m, t_sd, im_m, im_sd) * 255);
        }
    }

    return output;

}