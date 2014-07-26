#ifndef POS_LOG_H
#define POS_LOG_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>

typedef std::ostream& (*StreamPointer) (std::ostream& os);

struct XMLlog {
public:
  XMLlog();
  XMLlog(std::ostream& stream);
  XMLlog(std::string filename);
  ~XMLlog();
  void begin(std::string node);
  void log(const std::string& msg);
  void end();
  static std::string encodeString(std::string source);
  static const char endl = '\n';
  size_t depth() const {return stack.size(); }

  std::ostream* stream;
private:
  std::vector<std::string> stack;  
  bool lets_close;
  void registerSignals();
};

XMLlog& operator<<(XMLlog& log, const std::string& msg);
XMLlog& operator<<(XMLlog& log, char ch);
XMLlog& operator<<(XMLlog& log, const char* str);
XMLlog& operator<<(XMLlog& log, StreamPointer func);
XMLlog& operator<<(XMLlog& log, double val);
XMLlog& operator<<(XMLlog& log, float val);
XMLlog& operator<<(XMLlog& log, int val);
XMLlog& operator<<(XMLlog& log, size_t val);
#endif
