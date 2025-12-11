#include "pipnet.h"

using Function::PIPNet;


PIPNet::PIPNet(const std::string &_engine_path) : BasicTRTHandler(_engine_path) {}

std::vector<float> PIPNet::prepare(const cv::Mat& mat)
{
    cv::Mat canva = mat.clone();
	cv::resize(canva, canva, cv::Size(size, size));
    normalize_inplace(canva, mean_vals, norm_vals, false);
    std::vector<float> result(size * size * 3);
    trans2chw(canva, result);
    return result;
}

void PIPNet::warmup()
{
	DBG_LOGI("PIPNet warm up start\n");
    cv::Mat mat = cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
	std::vector<float> cur_input = this->prepare(mat);

	CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

	context->executeV2(buffers);

	float *outputs_cls = new float[1*68*8*8];
	float *outputs_x = new float[1*68*8*8];
	float *outputs_y = new float[1*68*8*8];
	float *outputs_nb_x = new float[1*68*10*8*8];
	float *outputs_nb_y = new float[1*68*10*8*8];

	CHECK(cudaMemcpy(outputs_cls, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outputs_x, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_y, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_x, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_y, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));

	delete outputs_cls;
	delete outputs_x;
	delete outputs_y;
	delete outputs_nb_x;
	delete outputs_nb_y;

    DBG_LOGI("PIPNet warm up done\n");
}

void PIPNet::predict(const cv::Mat &mat, cv::Rect_<int> img_rect, std::vector<cv::Point2i> &landmarks)
{
	if (mat.empty()) return;
	
	float img_height = static_cast<float>(mat.rows);
	float img_width = static_cast<float>(mat.cols);

	// 创建输入
	std::vector<float> cur_input = this->prepare(mat);

	// 将输入传递到GPU
    CHECK(cudaMemcpy(buffers[0], cur_input.data(), buffer_size[0], cudaMemcpyHostToDevice));

    // 异步执行
    context->executeV2(buffers);

	float *outputs_cls = new float[1*68*8*8];
	float *outputs_x = new float[1*68*8*8];
	float *outputs_y = new float[1*68*8*8];
	float *outputs_nb_x = new float[1*68*10*8*8];
	float *outputs_nb_y = new float[1*68*10*8*8];

	CHECK(cudaMemcpy(outputs_cls, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outputs_x, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_y, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_x, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_y, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));

	// 3. generate landmarks
	this->generate_landmarks(landmarks,
							 outputs_cls,
							 outputs_x,
							 outputs_y,
							 outputs_nb_x,
							 outputs_nb_y,
							 img_rect, img_height, img_width);

	delete[] outputs_cls;
	delete[] outputs_x;
	delete[] outputs_y;
	delete[] outputs_nb_x;
	delete[] outputs_nb_y;
}

void PIPNet::predictGPU(const cv::cuda::GpuMat& gpu_mat, cv::Rect_<int> img_rect, std::vector<cv::Point2i>& landmarks)
{
	if (gpu_mat.empty()) return;
	
	float img_height = static_cast<float>(gpu_mat.rows);
	float img_width = static_cast<float>(gpu_mat.cols);

	// 1. GPU resize到256x256（使用gpuResize，和CPU结果一致）
	if (gpu_resized_.rows != size || gpu_resized_.cols != size) {
		gpu_resized_.create(size, size, CV_8UC3);
	}
	gpuResize(gpu_mat.ptr<unsigned char>(), gpu_resized_.ptr<unsigned char>(),
			  gpu_mat.cols, gpu_mat.rows, size, size,
			  3, gpu_mat.step, gpu_resized_.step, nullptr);

	// 2. GPU预处理（normalize + HWC→CHW），直接写入buffers[0]
	//    注意：PIPNet的mean/norm是RGB顺序，输入是BGR，已转换为BGR顺序
	gpu_preprocessor_.process(gpu_resized_, (float*)buffers[0], size, mean_vals_bgr, norm_vals_bgr);

	// 3. TensorRT推理
	context->executeV2(buffers);

	// 4. D2H传输（保留）
	float *outputs_cls = new float[1*68*8*8];
	float *outputs_x = new float[1*68*8*8];
	float *outputs_y = new float[1*68*8*8];
	float *outputs_nb_x = new float[1*68*10*8*8];
	float *outputs_nb_y = new float[1*68*10*8*8];

	CHECK(cudaMemcpy(outputs_cls, buffers[1], buffer_size[1], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_x, buffers[2], buffer_size[2], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_y, buffers[3], buffer_size[3], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_x, buffers[4], buffer_size[4], cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outputs_nb_y, buffers[5], buffer_size[5], cudaMemcpyDeviceToHost));

	// 5. CPU后处理（保留）
	this->generate_landmarks(landmarks,
							 outputs_cls, outputs_x, outputs_y,
							 outputs_nb_x, outputs_nb_y,
							 img_rect, img_height, img_width);

	delete[] outputs_cls;
	delete[] outputs_x;
	delete[] outputs_y;
	delete[] outputs_nb_x;
	delete[] outputs_nb_y;
}

void PIPNet::generate_landmarks(std::vector<cv::Point2i> &landmarks,
								const float *outputs_cls_ptr,
								const float *outputs_x_ptr,
								const float *outputs_y_ptr,
								const float *outputs_nb_x_ptr,
								const float *outputs_nb_y_ptr,
								cv::Rect_<int> img_rect,
								float img_height, float img_width)
{
	const unsigned int grid_h = 8; // 8
	const unsigned int grid_w = 8; // 8
	const unsigned int grid_length = grid_h * grid_w;		  // 8 * 8 = 64
	const unsigned int input_h = 256;
	const unsigned int input_w = 256;

	// // fetch data from pointers
	// const float *outputs_cls_ptr = outputs_cls.GetTensorMutableData<float>();
	// const float *outputs_x_ptr = outputs_x.GetTensorMutableData<float>();
	// const float *outputs_y_ptr = outputs_y.GetTensorMutableData<float>();
	// const float *outputs_nb_x_ptr = outputs_nb_x.GetTensorMutableData<float>();
	// const float *outputs_nb_y_ptr = outputs_nb_y.GetTensorMutableData<float>();

	// find max_ids
	std::vector<unsigned int> max_ids(num_lms);
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		const float *score_ptr = outputs_cls_ptr + i * grid_length;
		unsigned int max_id = 0;
		float max_score = score_ptr[0];
		for (unsigned int j = 0; j < grid_length; ++j)
		{
			if (score_ptr[j] > max_score)
			{
				max_score = score_ptr[j];
				max_id = j;
			}
		}
		max_ids[i] = max_id; // range 0~64
	}
	// find x & y offsets
	std::vector<float> output_x_select(num_lms);
	std::vector<float> output_y_select(num_lms);
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		const float *offset_x_ptr = outputs_x_ptr + i * grid_length;
		const float *offset_y_ptr = outputs_y_ptr + i * grid_length;
		const unsigned int max_id = max_ids.at(i);
		output_x_select[i] = offset_x_ptr[max_id];
		output_y_select[i] = offset_y_ptr[max_id];
	}

	// find nb_x & nb_y offsets
	std::unordered_map<unsigned int, std::vector<float>> output_nb_x_select;
	std::unordered_map<unsigned int, std::vector<float>> output_nb_y_select;
	// initialize offsets map
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		std::vector<float> nb_x_offset(num_nb);
		std::vector<float> nb_y_offset(num_nb);
		output_nb_x_select[i] = nb_x_offset;
		output_nb_y_select[i] = nb_y_offset;
	}
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		for (unsigned int j = 0; j < num_nb; ++j)
		{
			const unsigned int max_id = max_ids.at(i);
			const float *offset_nb_x_ptr = outputs_nb_x_ptr + (i * num_nb + j) * grid_length;
			const float *offset_nb_y_ptr = outputs_nb_y_ptr + (i * num_nb + j) * grid_length;
			output_nb_x_select[i][j] = offset_nb_x_ptr[max_id];
			output_nb_y_select[i][j] = offset_nb_y_ptr[max_id];
		}
	}

	// calculate coords
	std::vector<float> lms_pred_x(num_lms);								// 68
	std::vector<float> lms_pred_y(num_lms);								// 68
	std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_x; // 68,10
	std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_y; // 68,10

	// initialize pred maps
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		std::vector<float> nb_x_offset(num_nb);
		std::vector<float> nb_y_offset(num_nb);
		lms_pred_nb_x[i] = nb_x_offset;
		lms_pred_nb_y[i] = nb_y_offset;
	}
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		float cx = static_cast<float>(max_ids.at(i) % grid_w);
		float cy = static_cast<float>(max_ids.at(i) / grid_w);
		// calculate coords & normalize
		lms_pred_x[i] = ((cx + output_x_select[i]) * (float)net_stride) / (float)input_w;
		lms_pred_y[i] = ((cy + output_y_select[i]) * (float)net_stride) / (float)input_h;
		for (unsigned int j = 0; j < num_nb; ++j)
		{
			lms_pred_nb_x[i][j] = ((cx + output_nb_x_select[i][j]) * (float)net_stride) / (float)input_w;
			lms_pred_nb_y[i][j] = ((cy + output_nb_y_select[i][j]) * (float)net_stride) / (float)input_h;
		}
	}

	// reverse indexes
	std::unordered_map<unsigned int, std::vector<float>> tmp_nb_x; // 68,max_len
	std::unordered_map<unsigned int, std::vector<float>> tmp_nb_y; // 68,max_len
	// initialize reverse maps
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		std::vector<float> tmp_x(max_len);
		std::vector<float> tmp_y(max_len);
		tmp_nb_x[i] = tmp_x;
		tmp_nb_y[i] = tmp_y;
	}
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		for (unsigned int j = 0; j < max_len; ++j)
		{
			unsigned int ri = reverse_index1[i * max_len + j];
			unsigned int rj = reverse_index2[i * max_len + j];
			tmp_nb_x[i][j] = lms_pred_nb_x[ri][rj];
			tmp_nb_y[i][j] = lms_pred_nb_y[ri][rj];
		}
	}

	// merge predictions
	for (unsigned int i = 0; i < num_lms; ++i)
	{
		float total_x = lms_pred_x[i];
		float total_y = lms_pred_y[i];
		for (unsigned int j = 0; j < max_len; ++j)
		{
			total_x += tmp_nb_x[i][j];
			total_y += tmp_nb_y[i][j];
		}
		float x = total_x / ((float)max_len + 1.f);
		float y = total_y / ((float)max_len + 1.f);
		x = std::min(std::max(0.f, x), 1.0f);
		y = std::min(std::max(0.f, y), 1.0f);

		landmarks.emplace_back(cv::Point2i(x * img_width + img_rect.x, 
										   y * img_height + img_rect.y));
	}
}
