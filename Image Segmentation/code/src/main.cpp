#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if(argc!=4){
        cout<<"Usage: ../seg input_image initialization_file output_mask"<<endl;
        return -1;
    }
    
    // Load the input image
    // the image should be a 3 channel image by default but we will double check that in the seam_carving
    Mat in_image;
    in_image = imread(argv[1]);
    
    
    if(!in_image.data)
    {
        cout<<"Could not load input image!!!!!!"<<endl;
        return -1;
    }
    
    if(in_image.channels()!=3){
        cout<<"Image does not have 3 channels!!! "<<in_image.depth()<<endl;
        return -1;
    }
    
    
    // the output image
    Mat out_image = in_image.clone();
    
    ifstream f(argv[2]);
    if(!f){
        cout<<"Could not load initial mask file!!!"<<endl;
        return -1;
    }
    
    int width = in_image.cols;
    int height = in_image.rows;
    cout<<"height"<<height<<endl;
    cout<<"width"<<width<<endl;
    
    int n;
    f>>n;
    vector <vector<int> > vecf(n, vector<int>(3, 0));
    vector <vector<int> > vecb(n, vector<int>(3, 0));
    vector<int> foreg;
    vector<int> backg;
    
    // print foreground and background vector
    for(int i=0;i<n;i++){
        for(int j=0;j<3;j++){
            cout<<vecf[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<"prev"<<endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<3;j++){
            cout<<vecb[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<"here"<<endl;
    int f1=0;
    int b1=0;
    // get the initil pixels
    for(int i=0;i<n;++i){
        int x, y, t;
        f>>x>>y>>t;
        cout<<"x.y.t"<<x<<y<<t<<endl;
        if(x<0 || x>=width || y<0 || y>=height){
            cout<<"I valid pixel mask!"<<endl;
            return -1;
        }
        Vec3b pixel;
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        if(t==1){
            pixel = in_image.at<Vec3b>(y, x);
            vecf[f1][0] = pixel.val[0];
            vecf[f1][1] = pixel.val[1];
            vecf[f1][2] = pixel.val[2];
            foreg.push_back((y * width) + x);
            f1++;
        } else {
            pixel = in_image.at<Vec3b>(y, x);
            vecb[b1][0] = pixel.val[0];
            vecb[b1][1] = pixel.val[1];
            vecb[b1][2] = pixel.val[2];
            backg.push_back((y * width) + x);
            b1++;
        }
    }
    // print foreground and background vector
    cout<<"f:"<<f1<<endl;
    cout<<"b:"<<b1<<endl;
    for(int i=0;i<f1;i++){
        for(int j=0;j<3;j++){
            cout<<vecf[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<"next"<<endl;
    for(int i=0;i<b1;i++){
        for(int j=0;j<3;j++){
            cout<<vecb[i][j]<<" ";
        }
        cout<<endl;
    }
    
    int total_pixel = 0;
    //read input image pixel by pixel
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            
            Vec3b pixel_in = in_image.at<Vec3b>(y, x);
            int min_f = INT_MAX;
            int min_b = INT_MAX;
            total_pixel++;
            
            // comparing pixel_in with each vector in vecf and get min
            for(int m=0;m<f1;m++){
                int a = abs(vecf[m][0] - pixel_in.val[0]);
                int b = abs(vecf[m][1] - pixel_in.val[1]);
                int c = abs(vecf[m][2] - pixel_in.val[2]);
                int totalf = a + b + c;
                if(totalf<min_f){
                    min_f = totalf;
                }
            }
            // comparing pixel_in with each vector in vecb and get min
            for(int n=0;n<b1;n++){
                int d = abs(vecb[n][0] - pixel_in.val[0]);
                int e = abs(vecb[n][1] - pixel_in.val[1]);
                int f = abs(vecb[n][2] - pixel_in.val[2]);
                int totalb = d + e + f;
                if(totalb<min_b){
                    min_b = totalb;
                }
            }
            
            Vec3b pixel_temp;
            pixel_temp.val[0] = 0;
            pixel_temp.val[1] = 0;
            pixel_temp.val[2] = 0;
            
            if(min_b<min_f){
                // pixel belongs to background  and vecb(black)
                pixel_temp.val[0] = 0;
                pixel_temp.val[1] = 0;
                pixel_temp.val[2] = 0;
            }else{
                // pixel belongs to foreground  and vecf(white)
                pixel_temp.val[0] = 255;
                pixel_temp.val[1] = 255;
                pixel_temp.val[2] = 255;
            }
            out_image.at<Vec3b>(y,x) = pixel_temp;
        }
    }
    
    vector < vector < pair < int, int > > > list_adj;
    list_adj.resize(total_pixel + 2);  //no of pixels + 2
    
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            Vec3b pixel_w = out_image.at<Vec3b>(y, x);
            int multi = 0;
            multi = width;
            int j = y*multi;
            int pixel_pos = j+x;
            //top left pixel
            if(y==0 && x==0){
                //        cout<<"inside loop 0"<<endl;
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                int weightr=0;
                int weightd=0;
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                    
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
            }
            //top row middle pixels
            else if(y==0 && x!=width-1 && x!=0){
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                int weightr=0;
                int weightd=0;
                int weightl=0;
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
            //left middle pixels
            else if(x==0 && y!=height-1 && y!=0){
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                int weightr=0;
                int weightd=0;
                int weightu=0;
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
            //bottom left pixel
            else if(x==0 && y==height-1){
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                int weightr=0;
                int weightu=0;
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
            //bottom middle pixels
            else if(y==height-1 && x!=width-1 && x!=0){
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                int weightr=0;
                int weightu=0;
                int weightl=0;
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
            }
            //bottom right pixel
            else if(y==height-1 && x==width-1){
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                int weightu=0;
                int weightl=0;
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
            //right middle pixel
            else if(x== width-1 && y!=0 && y!=height-1){
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                int weightu=0;
                int weightl=0;
                int weightd=0;
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
            }
            //top right pixel
            else if(y==0 && x==width-1){
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                int weightl=0;
                int weightd=0;
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
            else{
                Vec3b pixel_u = out_image.at<Vec3b>(y-1, x); //pixel up
                Vec3b pixel_l = out_image.at<Vec3b>(y, x-1); //pixel left
                Vec3b pixel_d = out_image.at<Vec3b>(y+1, x); //pixel down
                Vec3b pixel_r = out_image.at<Vec3b>(y, x+1); //pixel right
                int weightu=0;
                int weightl=0;
                int weightd=0;
                int weightr=0;
                if(pixel_w == pixel_u){
                    weightu = 55555555;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightu = 1;
                    pair < int, int > cpair = make_pair((y-1) * multi + x, weightu);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_l){
                    weightl = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightl = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x-1), weightl);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_d){
                    weightd = 55555555;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightd = 1;
                    pair < int, int > cpair = make_pair((y+1) * multi + x, weightd);
                    list_adj[pixel_pos].push_back(cpair);
                }
                if(pixel_w == pixel_r){
                    weightr = 55555555;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }else{
                    weightr = 1;
                    pair < int, int > cpair = make_pair(y * multi + (x+1), weightr);
                    list_adj[pixel_pos].push_back(cpair);
                }
                
            }
        }
    }
    cout<<"Weights Assigned"<<endl;
    int source_m = total_pixel ;  // main source
    int target_m = total_pixel + 1;  // main target
    
    for(int j=0; j<f1; j++){                // make edges from main source to all the given sources
        int weightm = 55555555;
        int s = foreg[j];
        pair < int, int > cpair = make_pair(s, weightm);
        list_adj[source_m].push_back(cpair);
        pair < int, int > dpair = make_pair(source_m, weightm);
        list_adj[s].push_back(dpair);
    }
    
    
    
    cout<<"Main source edges assigned"<<endl;
    for (int n=0; n<b1; n++){               // make edges from main target to all the given targets
        int weightm = 55555555;
        int t = backg[n];
        pair < int, int > cpair = make_pair(t, weightm);
        list_adj[target_m].push_back(cpair);
        pair < int, int > dpair = make_pair(target_m, weightm);
        list_adj[t].push_back(dpair);
    }
	cout<<"Main target edges assigned"<<endl;
    
	cout<<"Making residual graph"<<endl;
    int id = 1;
    while(id==1){
        id=0;
        int  min_flow = INT_MAX;
        int t_size = total_pixel+2;
        int* v = NULL;			 //array to track visited nodes
        v = new int[t_size];
        for(int k=0; k<t_size; k++){
            v[k] = 0;
        }
        queue < int > queue_bfs;  //bfs queue
        v[source_m] = 1; //start from main source and mark as visited
        queue_bfs.push(source_m); // push source to the bfs queue
        int pre[t_size];
        for(int k=0; k<t_size; k++){
            pre[k] = 0;
        }
        pre[source_m] = -1;
        for(;!queue_bfs.empty() && v[target_m]==0;){
            int w = queue_bfs.front();
            queue_bfs.pop();
            for(int i = 0; i<list_adj[w].size(); i++){
                if(v[list_adj[w][i].first]==0 && list_adj[w][i].second>0){
                    queue_bfs.push(list_adj[w][i].first);
                    pre[list_adj[w][i].first] = w;
                    v[list_adj[w][i].first] = 1;
                }
            }
            if(v[target_m]==1){
                break;
            }
        } //inner for loop
        
        if(v[target_m]==1){
            for(int p=target_m; p!=source_m; p=pre[p]){   //find minimum weight on the path
                int l= pre[p];
                int flow;
                for(int i = 0; i<list_adj[l].size();i++){
                    if (list_adj[l][i].first == p){
                        flow = list_adj[l][i].second; // weight of edge from l to p
                    }
                }
                if(flow<min_flow){
                    min_flow = flow;
                }
            }
    
            for(int p=target_m; p!=source_m; p=pre[p]){
                int l= pre[p];
                for(int g=0; g< list_adj[l].size();g++){
                    if (list_adj[l][g].first == p){
                        list_adj[l][g].second = list_adj[l][g].second - min_flow; //update residual graph in forward direction
                    }
                } //g
                for(int h=0; h< list_adj[p].size();h++){
                    if(list_adj[p][h].first == l){
                        list_adj[p][h].second = list_adj[p][h].second + min_flow; //update residual graph in backward direction
                    }
                } //h
            } //p
        } //if target visited
        id = 1;
        if(v[target_m]==0){
            break;
        }
    } //outer while loop

	cout<<"Applying Min-cut"<<endl;
    
    for(int y=0; y<height; y++){           // entire image as background
        for(int x=0; x<width; x++){
            Vec3b pixel_b;
            pixel_b.val[0] = 0;
            pixel_b.val[1] = 0;
            pixel_b.val[2] = 0;
            out_image.at < Vec3b > (y, x) = pixel_b;
            
        }
    }
    
    int t_size = total_pixel+2;
    queue<int> final_queue;
    int check[t_size];                   //array to track visited nodes
    for(int k=0; k<t_size; k++){
        check[k] = 1;
    }
    int s = source_m;
    final_queue.push(s);
    int cols = width;
    int y_s = s / cols;
    int x_s = s % cols;
    Vec3b pixel_f;
    pixel_f.val[0] = 255;
    pixel_f.val[1] = 255;
    pixel_f.val[2] = 255;
    out_image.at < Vec3b > (y_s, x_s) = pixel_f;
    for(;!final_queue.empty();){
        int fin = final_queue.front();
        final_queue.pop();
        for(int v=0; v<list_adj[fin].size(); v++){
            if(check[list_adj[fin][v].first]==1 && list_adj[fin][v].second>0){
                final_queue.push(list_adj[fin][v].first);
                int y_s = (list_adj[fin][v].first) / cols;
                int x_s = (list_adj[fin][v].first) % cols;
                out_image.at < Vec3b > (y_s, x_s) = pixel_f;
                check[list_adj[fin][v].first]=2;
            }
        }
    }
    
    cout<<"checkp3"<<endl;
    // write it on disk
    imwrite( argv[3], out_image);
    
    // also display them both
    
    namedWindow( "Original image", WINDOW_AUTOSIZE );
    namedWindow( "Show Marked Pixels", WINDOW_AUTOSIZE );
    imshow( "Original image", in_image );
    imshow( "Show Marked Pixels", out_image );
    waitKey(0);
    return 0;
}
