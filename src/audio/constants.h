#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>

namespace live{

	const short PCM_FORMAT_FLAG = 0x10;
	const std::string DATA_SIGN = "RIFF";
	const std::string WAV_SIGN = "WAVE";
	const std::string FMT_SIGN = "FMT";

	const int PCM_HEADER_LEN = 44;

	enum RESULT_TYPE {AUTO_PITCH,};

	const int ORDER = 4;

	const float FirstFilters[ORDER+1]={
		0.0006f,
		0.0026f,
		0.0038f,
		0.0026f,
		0.0006f};

	const float SecondFilters[ORDER+1]={
		1.0000f,
		-3.0787f,
		3.6372f,
		-1.9432f,
		0.3948f};

	const float CLIP = 0.68f;

	const int FrameLength = 480;
	const int FrameShift = 320;

	const int LowFrecuncy = 60;
	const int HighFrecuncy = 500;

	const int LpcOrder = 14;

	const double PI = 3.14159265358979323846;

	const double filterMax = 1E37;
}

#endif //CONSTANTS_H
