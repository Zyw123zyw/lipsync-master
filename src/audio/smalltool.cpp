#include "smalltool.h"
#include <cctype>
#include <algorithm>
#include <vector>
#include <functional>
#include <iterator>
using namespace std;

namespace live
{

bool Utility::doCompare(const string &s1, const string &s2){
	typedef pair<string::const_iterator, string::const_iterator> SItrs;
	SItrs its = mismatch(s1.begin(), s1.end(), s2.begin(), compareCharInsensitive);
	if(its.first == s1.end() && its.second == s2.end()){
		return true;
	}
	else{
		return false;
	}
}
bool Utility::compareCharInsensitive(char c1, char c2){
	return toupper(static_cast<unsigned char>(c1)) < toupper(static_cast<unsigned char>(c2));
}

bool Utility::compareStringInsensitive(const string &s1, const string &s2){
	if(s1.size() != s2.size())
		return false;
	return doCompare(s1, s2);
}

float Utility::getMedian(float *data, int count){
	vector<float> temp;
	copy(data, data + count, back_inserter(temp));

	sort(temp.begin(), temp.end(), greater<float>());

	return temp[(count - 1) / 2];
}


}