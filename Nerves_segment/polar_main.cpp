#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core.hpp>
#include <numeric>
#include <iostream>
#include <maxflow.h>


// Define structs
struct MouseParams {
    cv::Mat* im;
    std::vector<cv::Point2f>* coordinates;
    std::vector<double>* intensities;
};

// Function headers
void onMouseClick(int event, int x, int y, int flags, void* params);
cv::Mat graph_cut(cv::Mat im);

int main() {
    // Set options
    bool display_recovered = true;
    bool display_transform = false;
    bool save_im = true;
    bool display_cut = true;


    // Set warp flags (Here semi log polar)
    int flags = cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG;

    // Open image data
    cv::Mat src;
    src = cv::imread("../nerves_part.tiff", cv::IMREAD_GRAYSCALE);
    std::string title = "Nerves";


     // Initialize mouse click variables to safe both position and intensity
    std::vector<cv::Point2f> center_points; 
    std::vector<double> intensities;
    MouseParams mp;
    mp.im = &src;
    mp.coordinates = &center_points;
    mp.intensities = &intensities;

    // Display image and let the user select center points
    int n_points = 1;
    cv::namedWindow(title, 1);
    cv::setMouseCallback(title, onMouseClick, (void*)&mp);
    cv::imshow(title, src);

    // Close image after n points are selected
    while(intensities.size() < n_points) {
        cv::waitKey(1);
    }
    cv::destroyAllWindows();
    
    

    // Define center point and radius
    cv::Point2f center = center_points.at(0); // center( (float)src.cols / 2, (float)src.rows / 2 );
    double max_radius = 37.2; // 0.7*cv::min(center.y, center.x);

    cv::Mat transformed_im;
    cv::warpPolar(src, transformed_im, cv::Size(),center, max_radius, flags); 

    // Display transformed image
    if (display_transform) {
        std::string title {"Logpolar representation of nerves"};
        cv::namedWindow(title, 1);
        cv::imshow(title, transformed_im);
        cv::waitKey(0);
    }
    
    if (save_im) {
        std::string filename = "wraped_im.png";
        cv::imwrite(filename, transformed_im);
    } 

    // Transform back

    cv::Mat tmp_im, recovered_im;
    tmp_im = graph_cut(transformed_im);
    if (display_cut) {
        std::string title {"Graph cut"};
        cv::namedWindow(title, 1);
        cv::imshow(title, tmp_im);
        cv::waitKey(0);
    }


    cv::warpPolar(tmp_im, recovered_im, src.size(), center, max_radius, flags + cv::WARP_INVERSE_MAP);
    
    // Display recovered image
    if (display_recovered) {
        std::string title {"Logpolar representation of nerves recovered"};
        cv::namedWindow(title, 1);
        cv::imshow(title, recovered_im);
        cv::waitKey(0);
    }


}

void onMouseClick(int event, int x, int y, int flags, void* params) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        MouseParams* mp = (MouseParams*)params;
        (*(mp->coordinates)).push_back(cv::Point(x,y));
        double intensity = double ( (*(mp->im)).at<uchar>(y,x) );
        (*(mp->intensities)).push_back(intensity);
    }
}

cv::Mat graph_cut(cv::Mat im) {
    bool outer; // Specify if inner or outer border
    cv::Mat t_im;
    cv::transpose(im, t_im);
    cv::Mat g_im;
    cv::Sobel(t_im, g_im, CV_64FC1, 0, 1);

    if (outer) {
        g_im = -g_im;
    }
    

    cv::Mat src;
    g_im.convertTo(src, CV_64FC1);
    
    std::string title = "Wrapped image";
    

    int col = src.cols;
    int row = src.rows;
    std::vector<int> dims = {col, row};

    std::vector<int> range_vec(col*row);
    std::iota(std::begin(range_vec), std::end(range_vec), 0);

    // Make on region cost
    cv::Mat sub_im1 = src.rowRange(1, row);
    cv::Mat sub_im2 = src.rowRange(0, row-1);
    cv::Mat diff_im = sub_im1 - sub_im2;

    cv::Mat w_on = cv::Mat(row, col, CV_64FC1, -1);
    diff_im.rowRange(0, diff_im.rows).copyTo(w_on.rowRange(1, row));
    
    // Now working untill line 124 in grid cut 

    // Create W_in (In this case just row(0) full of inf)
    int inf = -999999;
    cv::Mat inf_tmp = cv::Mat(row, col, CV_32FC1, inf);
    inf_tmp.rowRange(0, 1).copyTo(w_on.rowRange(0, 1));

    // Generate graph
    maxflow::Graph_III g(range_vec.size(),1);
    g.add_node(range_vec.size());

    // Add terminal weights with the same id's as in the Matlab script
    int idx = 0;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (w_on.at<double>(r, c) < 0) {
                g.add_tweights(idx, -w_on.at<double>(r, c), 0);
                //std::cout << idx << ", " << -w_on.at<double>(r, c) << ", " << 0 << std::endl;
            }
            else {
                g.add_tweights(idx, 0, w_on.at<double>(r, c));
                // std::cout << idx << ", " << 0 << ", " << w_on.at<double>(r, c) << std::endl;
            }
            idx++;
        }
       
    }

    // Terminal weights added correctly. Only edgewights E from matlab missing
    for (int i = col; i < col*row; i++) {
        g.add_edge(i, i-col, -inf, 0);
        // std::cout << i << ", " << i-col << ", " << -inf << ", " << 0 << std::endl;
    }

    int smooth = 3;
    // Erpm (erxp)
    for (int i = 1; i < col*row-smooth*col; i++) {
        //std::cout << 4*col+i-1 << "  " << i << std::endl;
        g.add_edge(smooth*col+i-1, i, -inf, 0);
    }

    // Erpm (erxm)
    for (int i = 0; i <= col*row-smooth*col-2; i++) {
        //std::cout << 4*col+i+1 << "  " << i << std::endl;
        g.add_edge(smooth*col+i+1, i, -inf, 0);
    }

    // Erb
    for (int i = 0; i < col-1; i++) {
        g.add_edge(i, i+1, -inf, -inf);
    }
  

    // Compute flow
    double flow = g.maxflow();
    
    std::vector<int> source;
    for (int i = 0; i < col*row; i++) {
        if (g.what_segment(i) == 1) {
            source.push_back(i);
        }
    }

    
    // Create index image
    
    cv::Mat S = cv::Mat(row, col, CV_8U, 0.0);
    idx = 0;
    int count = 0;
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            if (count < source.size()) {
                if (idx == source.at(count)) {
                count++;
                S.at<uchar>(r, c) = 255;
                }
            }
            idx++;
        }
    }
   
   cv::Mat color_im;
   cv::cvtColor(t_im, color_im, cv::COLOR_GRAY2BGR);

    std::vector<cv::Point> contour;
 


    for (int i = 0; i < col; i++) {
        auto current_col = S.col(i);
        for(int j = 0; j < row; j++) {
            if (int(current_col.at<uchar>(j)) == 255) {
                contour.push_back(cv::Point(i,j));
                break;
            }
        }
    }
    const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
	int npts = cv::Mat(contour).rows;
	
	// draw the line

	polylines(color_im, &pts,&npts, 1,
	    		false, 			// draw closed contour (i.e. joint end to start) 
	            cv::Scalar(0,255,0),// colour RGB ordering (here = green) 
	    		1, 		        // line thickness
			    1, 0);

    cv::Mat output;
    cv::transpose(color_im, output);

    return output;

}