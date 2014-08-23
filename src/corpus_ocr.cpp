#include "corpus.h"
#include "boost/algorithm/string.hpp"
#include "boost/foreach.hpp"
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include "feature.h"

using namespace std;
using namespace boost;
using namespace Tagging;

template<size_t height, size_t width>
TokenOCR<height, width>::TokenOCR() {}

template<size_t height, size_t width>
TokenOCR<height, width>::TokenOCR(const string& line) : Token(line) {
}

template<size_t height, size_t width>
void TokenOCR<height, width>::parseline(const string& line) {
  vector<string> parts;
  split(parts, line, is_any_of(" "));
  tag = parts[1];
  fold = int(parts[5][0]);
  for(size_t i = 0; i < height; i++) {
    for(size_t j = 0; j < width; j++) {
      img[i][j] = int(parts[6 + i * width + j][0]);
    }
  }
}

template<size_t height, size_t width>
string TokenOCR<height, width>::str() const {
  string str = "";
  for(size_t i = 0; i < height; i++) {
    for(size_t j = 0; j < width; j++) {
      str += '0'+img[i][j];
    }
  }
  return str;
}

template<size_t height, size_t width>
void SentenceOCR<height, width>::parselines(const vector<string>& lines) {
  this->seq.clear();
  for(const string& line : lines) {
    ptr<TokenOCR<height, width> > token = ptr<TokenOCR<height, width> >(new TokenOCR<height, width>(line));
    this->seq.push_back(token);
  }
}

template<size_t height, size_t width> 
string SentenceOCR<height, width>::str() const {
  stringstream ss;
  for(const TokenPtr token : seq) {
    ss << token->str() << "/" << token->tag << "\t";
  }
  return ss.str();
}


