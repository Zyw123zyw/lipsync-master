#ifndef WAVDATA_H
#define WAVDATA_H

#include <vector>
#include <cassert>

namespace live
{
	class WavHelper;
	class WavData{
	public:
		WavData(){}
		inline long wavLength() const{
			return length;
		} 

		inline long sampleRate() const{
			return rate;
		}

		inline short sampleBitNumber() const{
			return bitNumber;
		}

		inline short at(const int &location) const{
			assert((location < length) && (location >= 0) && "location is out of range!");
			return wavVector[location];
		}

		inline short operator[](const int &location) const{
			return wavVector[location];
		}

		//maybe return a copy of the data is much safer!
		//but now I just return an handle, so you should be careful not to modify the data!
		//I may change the policy later!
		inline const short *data() const{
			return &wavVector[0];
		}

		//WavData(const WavData &wavData) : wavVector(wavData.wavVector){
		//	length = wavData.length;
		//	rate = wavData.rate;
		//	bitNumber = wavData.bitNumber;
		//}

	private:
		long rate;
		short bitNumber;
		int length;
		std::vector<short> wavVector;
		//we don't need these functions now!
		WavData &operator=(const WavData &);
		WavData(const WavData &);

		friend class WavHelper;
	};
}

#endif //WAVDATA_H
