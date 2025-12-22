#ifndef SMALLTOOL_H
#define SMALLTOOL_H
#include <string>


namespace live
{
	class Utility{
	public:
		static bool compareCharInsensitive(char, char);
		static bool compareStringInsensitive(const std::string &, const std::string &);
		static float getMedian(float *, int);
	private:
		static bool doCompare(const std::string&, const std::string&);
	};
}

#endif //SMALLTOOL_H
