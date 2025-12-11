#include "trt_utils.h"
#include <opencv2/imgproc/types_c.h>


void trans2chw(const cv::Mat &mat, std::vector<float> &input_tensor) {
    const int rows = mat.rows;
    const int cols = mat.cols;
    const int channels = mat.channels();

    cv::Mat mat_ref;
    if (mat.type() != CV_32FC(channels)) {
        mat.convertTo(mat_ref, CV_32FC(channels));
    } else {
        mat_ref = mat;
    }

    std::vector<cv::Mat> mats(channels);
    cv::split(mat_ref, mats);

    auto data = input_tensor.data();
    for (int i = 0; i < channels; ++i) {
        memcpy(data, mats[i].data, rows * cols * sizeof(float));
        data += rows * cols;
    }
}


void normalize_inplace(cv::Mat &mat_inplace,
                       const float *mean,
                       const float *norm,
                       bool swapRB)
{
    if (mat_inplace.type() != CV_32FC3)
        mat_inplace.convertTo(mat_inplace, CV_32FC3);
    for (unsigned int i = 0; i < mat_inplace.rows; ++i)
    {
        cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
        if (swapRB)
        {
            float tmp = 0.f;
            for (unsigned int j = 0; j < mat_inplace.cols; ++j)
            {
                tmp = p[j][0];
                p[j][0] = p[j][2];
                p[j][2] = tmp;
                p[j][0] = (p[j][0] - mean[0]) * norm[0];
                p[j][1] = (p[j][1] - mean[1]) * norm[1];
                p[j][2] = (p[j][2] - mean[2]) * norm[2];
            }
        }
        else
        {
            for (unsigned int j = 0; j < mat_inplace.cols; ++j)
            {
                p[j][0] = (p[j][0] - mean[0]) * norm[0];
                p[j][1] = (p[j][1] - mean[1]) * norm[1];
                p[j][2] = (p[j][2] - mean[2]) * norm[2];
            }
        }
    }
}

cv::Mat meanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32F);
    for (int i = 0; i < dim; i++)
    {
        float sum = 0;
        for (int j = 0; j < num; j++)
        {
            sum += src.at<float>(j, i);
        }
        output.at<float>(0, i) = sum / num;
    }

    return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B)
{
    cv::Mat output(A.rows, A.cols, A.type());

    assert(B.cols == A.cols);
    if (B.cols == A.cols)
    {
        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < B.cols; j++)
            {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }
    return output;
}

cv::Mat varAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return meanAxis0(temp_);
}

int MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}

cv::Mat similarTransform(cv::Mat src, cv::Mat dst)
{
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0)
    {
        d.at<float>(dim - 1, 0) = -1;
    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    int rank = MatrixRank(A);
    if (rank == 0)
    {
        assert(rank == 0);
    }
    else if (rank == dim - 1)
    {
        if (cv::determinant(U) * cv::determinant(V) > 0)
        {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        }
        else
        {
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else
    {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp;       // U
        T.rowRange(0, dim).colRange(0, dim) = U * diag_ * V;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim) * src_mean.t();
    cv::Mat temp2 = scale * temp1;
    cv::Mat temp3 = dst_mean - temp2.t();
    T.at<float>(0, 2) = temp3.at<float>(0);
    T.at<float>(1, 2) = temp3.at<float>(1);
    T.rowRange(0, dim).colRange(0, dim) *= scale; // T[:dim, :dim] *= scale

    return T;
}

//Perform inverse of complex matrix.
cv::Mat invComplex(const cv::Mat &m)
{
    //Create matrix with twice the dimensions of original
    cv::Mat twiceM(m.rows * 2, m.cols * 2, CV_MAKE_TYPE(m.type(), 1));

    //Separate real & imaginary parts
    std::vector<cv::Mat> components;
    cv::split(m, components);

    cv::Mat real = components[0], imag = components[1];

    //Copy values in quadrants of large matrix
    real.copyTo(twiceM({0, 0, m.cols, m.rows}));                //top-left
    real.copyTo(twiceM({m.cols, m.rows, m.cols, m.rows}));      //bottom-right
    imag.copyTo(twiceM({m.cols, 0, m.cols, m.rows}));           //top-right
    cv::Mat(-imag).copyTo(twiceM({0, m.rows, m.cols, m.rows})); //bottom-left

    //Invert the large matrix
    cv::Mat twiceInverse = twiceM.inv();

    cv::Mat inverse(m.cols, m.rows, m.type());

    //Copy back real & imaginary parts
    twiceInverse({0, 0, inverse.cols, inverse.rows}).copyTo(real);
    twiceInverse({inverse.cols, 0, inverse.cols, inverse.rows}).copyTo(imag);

    //Merge real & imaginary parts into complex inverse matrix
    cv::merge(components, inverse);
    return inverse;
}


void normalize_kp_inline(std::vector<cv::Point2f> &kp)
{   
    float x_mean = kp[kp.size()-1].x;
    float y_mean = kp[kp.size()-1].y;
    kp.pop_back();

    for (size_t i = 0; i < kp.size(); i++)
    {
        kp[i].x = kp[i].x - x_mean;
        kp[i].y = kp[i].y - y_mean;
    }
    std::vector<cv::Point2f> kp_hull;
    cv::convexHull(kp, kp_hull);
    double area = cv::contourArea(kp_hull);
    area = (float)std::sqrt(area);

    for (size_t i = 0; i < kp.size(); i++)
    {
        kp[i].x = kp[i].x / area;
        kp[i].y = kp[i].y / area;
    }
    
}


float cal_norm(std::vector<cv::Point2f> kp_source, std::vector<cv::Point2f> kp_driving)
{
    float sum = 0;
    for (size_t i = 36; i < 68; i++)
    {
        float tmp = std::pow(std::abs(kp_source[i].x - kp_driving[i].x), 2) + std::pow(std::abs(kp_source[i].y - kp_driving[i].y), 2);
        sum = sum + tmp;
    }
    return sum;
}


void mean_point(std::vector<cv::Point2f> &landmarks)
{
    float x_mean, y_mean;
    float x_total = 0;
    float y_total = 0;
    for (size_t i = 0; i < 68; i++)
    {
        x_total = x_total + landmarks[i].x;
        y_total = y_total + landmarks[i].y;
    }
    x_mean = x_total / 68;
    y_mean = y_total / 68;
    landmarks.push_back(cv::Point2f(x_mean, y_mean));
}

std::string rand_str(const int len) /*参数为字符串的长度*/
{
    /*初始化*/
    std::string str; /*声明用来保存随机字符串的str*/
    char c;          /*声明字符c，用来保存随机生成的字符*/
    int idx;         /*用来循环的变量*/
    srand(clock());
    /*循环向字符串中添加随机生成的字符*/
    for (idx = 0; idx < len; idx++)
    {
        /*rand()%26是取余，余数为0~25加上'a',就是字母a~z,详见asc码表*/
        c = 'a' + rand() % 26;
        str.push_back(c); /*push_back()是string类尾插函数。这里插入随机字符c*/
    }
    return str; /*返回生成的随机字符串*/
}

bool isFileExists_stat(std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}


int devide(std::string video_path, std::string& voise_path)
{
    int backslashIndex = video_path.find_last_of('/');
    std::string base_path = video_path.substr(0, backslashIndex);

    std::string voise_name = rand_str(13) + ".aac";
    voise_path = base_path + "/" + voise_name;

    // ffmpeg -i 3.mp4 -vn -y -acodec copy 3.aac
    std::string command_rd = "ffmpeg -loglevel 16 -i " + (std::string)video_path + " -vn -y -acodec copy -ar 48000 " + voise_path;
    DBG_LOGI("devide command:  %s\n", command_rd.c_str());
    system(command_rd.c_str());

    if (isFileExists_stat(voise_path))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


int getvoise(std::string dvideo_path, std::string dvoise_path)
{
    // ffmpeg -i 3.mp4 -vn -y -acodec copy 3.aac
    std::string command_rd = "ffmpeg -loglevel 16 -i " + (std::string)dvideo_path + " -vn -y -acodec copy -ar 48000 " + (std::string)dvoise_path;
    DBG_LOGI("devide command:  %s\n", command_rd.c_str());
    system(command_rd.c_str());

    if (isFileExists_stat(dvoise_path))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int merge(std::string video_path, std::string voise_path, std::string output_path)
{
    // ffmpeg -i video2.avi -i audio.mp3 -vcodec copy -acodec copy output.avi -bsf:a aac_adtstoasc
    std::string command_rd = "ffmpeg -loglevel 16 -i " + video_path + " -i " + voise_path + " -vcodec copy -acodec copy -bsf:a aac_adtstoasc " + output_path + " -y";
    DBG_LOGI("merge command:  %s\n", command_rd.c_str());
    system(command_rd.c_str());
    if (isFileExists_stat(output_path))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


void pick_rect(cv::Rect_<int> &g_box, std::vector<cv::Rect_<int>> &t_boxes, 
               cv::Rect_<int> &p_box, float threshold)
{
    if (t_boxes.size() == 0)
        return;
    float max_iou = 0;
    int best_ind = 0;
    for (size_t ind = 0; ind < t_boxes.size(); ind++)
    {
        cv::Rect_<int> inter = g_box & t_boxes[ind];
        int inter_area = inter.area();
        int small_box_area = std::min(g_box.area(), t_boxes[ind].area());
        float iou = inter_area * 1.0 / small_box_area;
        if (iou > max_iou){
            max_iou = iou;
            best_ind = ind;
        }    
    }
    if (max_iou < threshold)
        return;
    p_box = t_boxes[best_ind];
}

void merge_mask(cv::Mat &source_mask, cv::Mat &drived_mask, cv::Mat &out_mask)
{
    out_mask = cv::Mat::zeros(source_mask.rows, source_mask.cols, CV_8UC1);
    for (int i = 0; i < out_mask.rows; i++)
    {
        for (int j = 0; j < out_mask.cols; j++)
        {
            if (source_mask.at<uchar>(i, j) == 255 || drived_mask.at<uchar>(i, j) == 255)
            {
                out_mask.at<uchar>(i, j) = 255;
            }
            else
            {
                out_mask.at<uchar>(i, j) = drived_mask.at<uchar>(i, j);
            }
            
        }
    }
    out_mask.convertTo(out_mask, CV_32FC1);
    out_mask /= 255.0;
}


void mask2dist(cv::Mat &mask, cv::Mat &dist, int gap, int pad)
{
    // 超参数pad控制歪头矫正时mask向内收缩的范围，头顶因为影响较大，不进行收缩
    dist = mask.clone();
    for (int i = 0; i < dist.rows; i++)
    {   
        for (int j = 0; j < dist.cols; j++)
        {            
            if (i > gap && mask.at<float>(i, j) > 0.5){
                dist.at<float>(i, j) = 1;
            }
            if (mask.at<float>(i, j) < 0.9){
                dist.at<float>(i, j) = 0;
            }
        }
    }

    dist *= 255;
    dist.convertTo(dist, CV_8UC1);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(dist, dist, element);
    dist.convertTo(dist, CV_32FC1, 1/255.f);

    cv::Mat bg_mask = cv::Scalar(1) - dist;
    bg_mask.convertTo(bg_mask, CV_8UC1);
    cv::distanceTransform(bg_mask, bg_mask, CV_DIST_L1, 5);
    
    for (int i = 0; i < dist.rows; i++)
    {   
        for (int j = 0; j < dist.cols; j++)
        {
            if (bg_mask.at<float>(i, j) == 0){
                dist.at<float>(i, j) = 1;
            }
            else{
                dist.at<float>(i, j) = 1.0 / bg_mask.at<float>(i, j);
            }

            if (i < 0 || i > dist.rows-pad)
            {
                dist.at<float>(i, j) = 0;
            }
            if (j < pad || j > dist.cols-pad)
            {
                dist.at<float>(i, j) = 0;
            }
        }
    }

    cv::GaussianBlur(dist, dist, cv::Size(7, 7), 0, 0);
}