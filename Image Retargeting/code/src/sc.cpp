#include "sc.h"
#include <math.h>
#include<stack>

using namespace cv;
using namespace std;


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){
    
    // some sanity checks
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! new height has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_width<=0){
        cout<<"Invalid request!!! new width has to be positive!"<<endl;
        return false;
        
    }
    if(new_height<=0){
        cout<<"Invalid request!!! new height has to be positive!"<<endl;
        return false;
    }
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}

// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){
    
    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
    
    while(iimage.rows!=new_height || iimage.cols!=new_width){
        // horizontal seam if needed
        if(iimage.rows>new_height){
            reduce_horizontal_seam_trivial(iimage, oimage);
            iimage = oimage.clone();
        }
        if(iimage.cols>new_width){
            reduce_vertical_seam_trivial(iimage, oimage);
            iimage = oimage.clone();
        }
    }
    out_image = oimage.clone();
    return true;
}

// horizontl trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& iimage, Mat& out_image){
    int width = iimage.cols;
    int height = iimage.rows;
    int density[height][width];
    
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            Vec3b pixel_w = iimage.at<Vec3b>(y, x);
            //top left pixel
            if(y==0 && x==0){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1,0); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(0, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //top row middle pixels
            else if(y==0 && x!=width-1 && x!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1,x); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //left middle pixels
            else if(x==0 && y!=height-1 && y!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom left pixel
            else if(x==0 && y==height-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(height-1, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(0, 0); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom middle pixels
            else if(y==height-1 && x!=width-1 && x!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(0, x); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom right pixel
            else if(y==height-1 && x==width-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(0, width-1); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, 0); //pixel right changed
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //right middle pixel
            else if(x== width-1 && y!=0 && y!=height-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, 0); //pixel right changed
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //top right pixel
            else if(y==0 && x==width-1){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1, width-1); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(0, 0); //pixel right changed
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            else{
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow(pixel_u.val[0]-pixel_d.val[0],2);
                double gx = pow(pixel_u.val[1]-pixel_d.val[1],2);
                double rx = pow(pixel_u.val[2]-pixel_d.val[2],2);
                double x_grad = bx + gx +rx;
                double by = pow(pixel_l.val[0]-pixel_r.val[0],2);
                double gy = pow(pixel_l.val[1]-pixel_r.val[1],2);
                double ry = pow(pixel_l.val[2]-pixel_r.val[2],2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
        } //x-for loop
    }//y-for loop
    
    for(int x=1; x<width; x++){
        for(int y=0; y<height; y++){
            if(y==0){ // first row
                density[y][x] = density[y][x] + (min( density[y][x-1] , density[y+1][x-1]));
            }//x=0
            else if(y== height-1){ //last row
                density[y][x] = density[y][x] + (min( density[y-1][x-1], density[y][x-1]) );
            }//x=width-1
            else{ //second row to second last row
                density[y][x] = density[y][x] + (min( density[y-1][x-1], (min (density[y][x-1] , density[y+1][x-1]))));
            }//else
            
        }
    }
    
    int min_last_col = density[0][width-1];
    //  int min_last_col = min_density[0][0];
    int min_row = 0;
    for(int y=0; y<height-1; y++){
        int x=width-1;
        if (density[y][x]< min_last_col){
            min_last_col = density[y][x];
            min_row = y;
        }
    }
    
    stack<int> pathc;
    pathc.push(min_row);
    int min_col_val; //minimum value of the column
    
    for(int x=width-1;x>0;x--){
        if(min_row == 0){
            min_col_val = min( density[min_row+1][x-1] , density[min_row][x-1]);
            if(min_col_val == density[min_row+1][x-1]){
                min_row = min_row+1;
            }
            pathc.push(min_row);
            
        }else if(min_row == height-1){
            min_col_val = min(density[min_row-1][x-1],density[min_row][x-1]);
            if(min_col_val == density[min_row-1][x-1]){
                min_row = min_row-1;
            }
            pathc.push(min_row);
            
        }else{
            min_col_val = min(density[min_row-1][x-1], (min(density[min_row][x-1],density[min_row+1][x-1])));
            if(min_col_val == density[min_row-1][x-1]){
                min_row = min_row-1;
            }else if(min_col_val == density[min_row+1][x-1]){
                min_row = min_row+1;
            }
            pathc.push(min_row);
        }
        
    }
    
    // retrieve the dimensions of the new image
    int rows_out = iimage.rows-1;
    int cols_out = iimage.cols;
    
    // create an image slighly smaller
    out_image = Mat(rows_out, cols_out, CV_8UC3);
    
    for(int x=0; x<width;x++){
        if(pathc.top() == height-1){
            for(int y=0; y<height-1; y++){
                
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y,x) = pixel;
            }
        }else{
            for(int y=0; y<pathc.top(); y++){
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y,x) = pixel;}
            for(int y=pathc.top()+1; y<height; y++){
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y-1,x) = pixel;}
        }
        pathc.pop();
    }//y
    
    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& iimage, Mat& out_image){
    int width = iimage.cols;
    int height = iimage.rows;
    int density[height][width];
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            Vec3b pixel_w = iimage.at<Vec3b>(y, x);
            //top left pixel
            if(y==0 && x==0){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1,0); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(0, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //top row middle pixels
            else if(y==0 && x!=width-1 && x!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1,x); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //left middle pixels
            else if(x==0 && y!=height-1 && y!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom left pixel
            else if(x==0 && y==height-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(height-1, width-1); //pixel left changed
                Vec3b pixel_d = iimage.at<Vec3b>(0, 0); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom middle pixels
            else if(y==height-1 && x!=width-1 && x!=0){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(0, x); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //bottom right pixel
            else if(y==height-1 && x==width-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(0, width-1); //pixel down changed
                Vec3b pixel_r = iimage.at<Vec3b>(y, 0); //pixel right changed
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //right middle pixel
            else if(x== width-1 && y!=0 && y!=height-1){
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, 0); //pixel right changed
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            //top right pixel
            else if(y==0 && x==width-1){
                Vec3b pixel_u = iimage.at<Vec3b>(height-1, width-1); //pixel up changed
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(0, 0); //pixel right changed
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
            else{
                Vec3b pixel_u = iimage.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = iimage.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = iimage.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = iimage.at<Vec3b>(y, x+1); //pixel right
                double bx = pow (pixel_u.val[0]-pixel_d.val[0], 2);
                double gx = pow (pixel_u.val[1]-pixel_d.val[1], 2);
                double rx = pow (pixel_u.val[2]-pixel_d.val[2], 2);
                double x_grad = bx + gx +rx;
                double by = pow (pixel_l.val[0]-pixel_r.val[0], 2);
                double gy = pow (pixel_l.val[1]-pixel_r.val[1], 2);
                double ry = pow (pixel_l.val[2]-pixel_r.val[2], 2);
                double y_grad = by + gy +ry;
                double energy = x_grad + y_grad;
                density[y][x] = abs(energy);
            }
        } //x-for loop
    }//y-for loop
    
    for(int y=1; y<height; y++){
        for(int x=0; x<width; x++){
            if(x==0){ // first col
                density[y][x] = density[y][x] + (min( density[y-1][x] , density[y-1][x+1]));
            }//x=0
            else if(x== width-1){ //last col
                density[y][x] = density[y][x] + (min( density[y-1][x-1] , density[y-1][x]) );
            }//x=width-1
            else{ //second col to second last col
                density[y][x] = density[y][x]  + (min( density[y-1][x-1] , (min (density[y-1][x] , density[y-1][x+1]))));
            }//else
            
        }
    }
    
    int min_last_row = density[height-1][0];
    int min_col = 0;
    for(int x=0; x<width-1; x++){
        int y=height-1;
        if (density[y][x]< min_last_row){
            min_last_row = density[y][x];
            min_col = x;
        }
    }
    
    stack<int> path;
    path.push(min_col);
    int min_row_val; //minimum value in the row
    
    for(int y=height-1;y>0;y--){
        if(min_col == 0){
            min_row_val = min(density[y-1][min_col],density[y-1][min_col+1]);
            if(min_row_val == density[y-1][min_col+1]){
                min_col = min_col+1;
            }
            path.push(min_col);
            
        }else if(min_col == width-1){
            min_row_val = min(density[y-1][min_col],density[y-1][min_col-1]);
            if(min_row_val == density[y-1][min_col-1]){
                min_col = min_col-1;
            }
            path.push(min_col);
            
        }else{
            min_row_val = min(density[y-1][min_col+1], (min(density[y-1][min_col],density[y-1][min_col-1])));
            if(min_row_val == density[y-1][min_col+1]){
                min_col = min_col+1;
            }else if(min_row_val == density[y-1][min_col-1]){
                min_col = min_col-1;
            }
            path.push(min_col);
        }
        
    }
    
    // retrieve the dimensions of the new image
    int rows_out = iimage.rows;
    int cols_out = iimage.cols-1;
    
    // create an image slighly smaller
    out_image = Mat(rows_out, cols_out, CV_8UC3);
    
    for(int y=0; y<height;y++){
        //  for(int x=0; x<width; x++){
        if(path.top() == width-1){
            for(int x=0; x<width-1; x++){
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y,x) = pixel;
            }
        }else{
            for(int x=0; x<path.top(); x++){
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y,x) = pixel;}
            for(int x=path.top()+1; x<width; x++){
                Vec3b pixel = iimage.at<Vec3b>(y, x);
                out_image.at<Vec3b>(y,x-1) = pixel;}
        }
        path.pop();
    }//y
    
    return true;
}
