#include "wavhelper.h"
#include "constants.h"
#include "smalltool.h"
#include <ios>
#include <fstream>
#include <algorithm>

using namespace std;


#include <sys/stat.h>

namespace live
{

struct stat st;
//here, i don't define a struct of wav header and analyze it
//instead i just read the header as an array of char and analyze using reinterpret_cast
//maybe this is not a good idea, but for now i just don't want to define more structs or classes

/*static*/ bool WavHelper::readWav(WavData &wav, const char *path)
{
    ifstream wavFile;
    wavFile.open(path, ios::binary);

    if (wavFile.fail()) {
        return false;
    }

    char wavHead[PCM_HEADER_LEN];
    wavFile.read(wavHead, PCM_HEADER_LEN);

#ifdef    NEED_CHECK
    if(Utility::compareStringInsensitive(string(wavHead, wavHead + 4), DATA_SIGN)){
        return false;
    }
    if(Utility::compareStringInsensitive(string(wavHead + 8, wavHead + 12), WAV_SIGN)){
        return false;
    }
#endif
    stat(path, &st);
    int fsz = st.st_size - 8;

    wav.rate = *(reinterpret_cast<int *>(&wavHead[24]));
    wav.bitNumber = *(reinterpret_cast<short *>(&wavHead[34]));
    int size = *(reinterpret_cast<int *>(&wavHead[40]));
    int size2 = *(reinterpret_cast<int *>(&wavHead[4]));
    if (size2 - 36 != size) size = size2 - 36;
    if (size & 0x01)
        size = size - 1;

    if (size > fsz) size = fsz; //wave length is determined by the file size

    wav.length = size / sizeof(short);

    short *tempData = new short[wav.length];
    wavFile.read(reinterpret_cast<char *>(tempData), size);
    wav.wavVector.clear();
    wav.wavVector.assign(tempData, tempData + wav.length);

    delete[] tempData;

    wavFile.close();
    return true;
}

//now this function has no use, just for future purpose!

/*static*/ bool WavHelper::writeWav(WavData &wav, const wstring &path) {
    return false;
}


}