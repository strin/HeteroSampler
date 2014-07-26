#include "log.h"
#include <ostream>
#include <fstream>
#include <boost/algorithm/string/replace.hpp>

using namespace std;

XMLlog::XMLlog() 
:stream(&cout), lets_close(false) {}

XMLlog::XMLlog(ostream& stream)
:stream(&stream), lets_close(false) {}

XMLlog::XMLlog(string filename) 
:stream(new fstream(filename.c_str())), lets_close(true) {}

XMLlog::~XMLlog() {
  if(lets_close) 
    delete stream;
}

void XMLlog::begin(std::string node) {
  boost::replace_all(node, " ", "_");
  *stream << "<" << node << ">" << endl;
  stack.push_back(node);
}

void XMLlog::end() {
  *stream << "</" << stack.back() << ">" << endl;
  stack.pop_back();
}

void XMLlog::log(const string& msg) {
  *stream << msg;
  (*stream).flush();
}

XMLlog& operator<<(XMLlog& log, const std::string& msg) {
  log.log(msg); 
  return log;
}

XMLlog& operator<<(XMLlog& log, const char* str) {
  log.log(string(str));
  return log;
}

XMLlog& operator<<(XMLlog& log, StreamPointer func) {
  log.stream = &(func(*log.stream));
  return log;
}
