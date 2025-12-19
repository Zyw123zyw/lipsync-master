#ifndef WAVHELPER_H
#define WAVHELPER_H

#include <string>
#include "wavdata.h"

namespace live
{
	class WavHelper{
	public:
		static bool readWav(WavData &, const char *);
		static bool writeWav(WavData &, const std::wstring &);
	};
}

#endif //WAVHELPER_H
