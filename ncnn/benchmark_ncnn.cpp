#include <net.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;

ncnn::Option opt;
ncnn::Layer *softmax;

std::vector<cv::Point> heatmap2lmks1(ncnn::Mat heatmap){
    ncnn::Mat score;
    score.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);
    ncnn::Mat index_quotient, index_remainder;
    index_quotient.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);
    index_remainder.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);

    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = heatmap.c/2; q < heatmap.c-1; q+=1)
    {
        float* ptr = &heatmap[(q)*64*64];
        float* ptr_score = &score[int(q-68)*9];
        float* ptr_index_quotient = &index_quotient[int(q-68)*9];
        float* ptr_index_remainder = &index_remainder[int(q-68)*9];
        std::vector<std::pair<float, int> > vec;
        vec.resize(64*64);
        for (int i = 0; i < 64*64; i++)
        {
            vec[i] = std::make_pair(ptr[i], i);
        }

        std::partial_sort(vec.begin(), vec.begin() + 9, vec.end(),
                        std::greater<std::pair<float, int> >());

        for (int i = 0; i < 9; i++)
        {
            ptr_score[i] = vec[i].first;
            ptr_index_quotient[i] = vec[i].second/64; //y
            ptr_index_remainder[i] = vec[i].second%64; //x
        }
    }

    softmax = ncnn::create_layer(ncnn::layer_to_index("Softmax"));
    ncnn::ParamDict softmax_param;
    softmax_param.set(0, int(1)); // axis
    softmax_param.set(1, int(1)); // fixbug
    std::cout << softmax_param.get(0, int(0)) << std::endl;
    softmax->load_param(softmax_param);
    softmax->forward_inplace(score, opt);

    ncnn::Mat x, y;
    x.create(68, heatmap.elemsize, opt.blob_allocator);
    y.create(68, heatmap.elemsize, opt.blob_allocator);


    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < heatmap.c/2; q++)
    {
        float* ptr_x = &x[q];
        float* ptr_y = &y[q];
        float* ptr_score = &score[q*9];
        float* ptr_index_quotient = &index_quotient[q*9];
        float* ptr_index_remainder = &index_remainder[q*9];
        float temp_x, temp_y;
        temp_x = 0;
        temp_y = 0;
        for (int i = 0; i < 9; i++)
        {
            temp_x += float(ptr_index_remainder[i]*ptr_score[i]);
            temp_y += float(ptr_index_quotient[i]*ptr_score[i]);
        }
        ptr_x[0] = 4*temp_x;
        ptr_y[0] = 4*temp_y;
    }


    // // pretty_print(x);
    std::vector<cv::Point> lmks;
    
    for (int i = 0; i < 68; i++) {
        lmks.push_back(cv::Point2f(x[i], y[i]));
    }
    return lmks;
}


std::vector<float> heatmap2lmks(ncnn::Mat heatmap, ncnn::Option opt, ncnn::Layer *softmax){
    ncnn::Mat score;
    score.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);
    ncnn::Mat index_quotient, index_remainder;
    index_quotient.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);
    index_remainder.create(9, heatmap.c/2, heatmap.elemsize, opt.blob_allocator);

//     #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = heatmap.c/2; q < heatmap.c-1; q+=1)
    {
        float* ptr = &heatmap[(q)*48*48];
        float* ptr_score = &score[int(q-106)*9];
        float* ptr_index_quotient = &index_quotient[int(q-106)*9];
        float* ptr_index_remainder = &index_remainder[int(q-106)*9];
        std::vector<std::pair<float, int> > vec;
        vec.resize(48*48);
        for (int i = 0; i < 48*48; i++)
        {
            vec[i] = std::make_pair(ptr[i], i);
        }

        std::partial_sort(vec.begin(), vec.begin() + 9, vec.end(),
                          std::greater<std::pair<float, int> >());

        for (int i = 0; i < 9; i++)
        {
            ptr_score[i] = vec[i].first;
            ptr_index_quotient[i] = vec[i].second/48; //y
            ptr_index_remainder[i] = vec[i].second%48; //x
        }
    }

    softmax = ncnn::create_layer(ncnn::layer_to_index("Softmax"));
    ncnn::ParamDict softmax_param;
    softmax_param.set(0, int(1)); // axis
    softmax_param.set(1, int(1)); // fixbug
    std::cout << softmax_param.get(0, int(0)) << std::endl;
    softmax->load_param(softmax_param);
    softmax->forward_inplace(score, opt);

    ncnn::Mat x, y;
    x.create(106, heatmap.elemsize, opt.blob_allocator);
    y.create(106, heatmap.elemsize, opt.blob_allocator);


//     #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < heatmap.c/2; q++)
    {
        float* ptr_x = &x[q];
        float* ptr_y = &y[q];
        float* ptr_score = &score[q*9];
        float* ptr_index_quotient = &index_quotient[q*9];
        float* ptr_index_remainder = &index_remainder[q*9];
        float temp_x, temp_y;
        temp_x = 0;
        temp_y = 0;
        for (int i = 0; i < 9; i++)
        {
            temp_x += float(ptr_index_remainder[i]*ptr_score[i]);
            temp_y += float(ptr_index_quotient[i]*ptr_score[i]);
        }
        ptr_x[0] = 4*temp_x;
        ptr_y[0] = 4*temp_y;
    }


    // // pretty_print(x);
    std::vector<float> lmks;
    std::vector<int> lmks_68_ind = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 58, 59, 60, 61, 62, 66, 67, 69, 70, 71, 73, 75, 76, 78, 79, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103};
    for (int i = 0; i < lmks_68_ind.size(); i++) {
        lmks.push_back(x[lmks_68_ind[i]]/192.0);
        lmks.push_back(y[lmks_68_ind[i]]/192.0);

    }
    return lmks;
}
int main(){
    ncnn::Net pfld;
    ncnn::Option opt;

    // PFlD GhostNet
    // pfld.load_param("../pfld_sim.param");
    // pfld.load_model("../pfld_sim.bin");

    // PFLD Mobilenet
    // pfld.load_param("../pfld_mv2_sim.param");
    // pfld.load_model("../pfld_mv2_sim.bin");
    pfld.load_param("../model.param");
    pfld.load_model("../model.bin");


    cv::Mat img = cv::imread("../test.png", 1);
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(192, 192));
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_RGB, img.cols, img.rows, 192, 192);
    

    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = pfld.create_extractor();
    int num_threads = 1;
    ex.set_num_threads(num_threads);

    ex.input("input_1", in);
    ncnn::Mat out1, out2, out3, out4;
    float sum = 0;
    float n = 0;
    auto t0=0;
    auto t1=0;
    std::vector<float> lmks;
    std::vector<cv::Point> lmks_point;

    for(int i =0 ; i<2; i++){
        t0 = cv::getTickCount();
        // ex.extract("386", out1);
        // ex.extract("397", out2);
        // ex.extract("408", out3);

        // std::cout<<"Size output1: "<<out1.w<<", "<<out1.h<<", "<<out1.c<<std::endl;
        // std::cout<<"Size output1: "<<out2.w<<", "<<out2.h<<", "<<out2.c<<std::endl;
        // std::cout<<"Size output1: "<<out3.w<<", "<<out3.h<<", "<<out3.c<<std::endl;

        // ex.input("409", out1);
        // ex.input("409", out2);
        // ex.input("409", out3);
        ex.extract("output_1", out4);

        std::cout<<"Size output1: "<<out4.w<<", "<<out4.h<<", "<<out4.c<<std::endl;


        ncnn::Layer *softmax;
        lmks =  heatmap2lmks(out4, opt, softmax);
        for (int j=0; j<68; j++){
            lmks_point.push_back(cv::Point(lmks[2*j]*192, lmks[2*j+1]*192));
        }
        // std::cout<<"Lmks shape : "<<lmks.size()<<std::endl;
        t1 = cv::getTickCount();
        float ms = (t1-t0)*1000.0/cv::getTickFrequency();
        cout<<"Run :"<<ms<<endl;
        sum += ms;
        n++;
       
        for (auto lmk : lmks_point) {
        // std::cout<<"Point : "<<lmk<<std::endl;
         cv::circle(img, lmk, 1, cv::Scalar(0, 255, 0), -1);
        }
        
        cv::imshow("Image: ", img);
        if (cv::waitKey(0)==27) cv::destroyAllWindows();
    }

    // for (auto lmk : lmks_point) {
    //     // std::cout<<"Point : "<<lmk<<std::endl;
    //     cv::circle(img, lmk, 1, cv::Scalar(0, 255, 0), -1);
    // }
   

    cout<<"time inference: "<<(sum)<<"(ms)"<<endl;
    return 0;
}
