#ifndef POS_LOG_H
#define POS_LOG_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <thread>
#include <map>
#include <list>
#include <unordered_map>
#include <mutex>

typedef std::ostream& (*StreamPointer) (std::ostream& os);

struct XMLlog;

XMLlog& operator<<(XMLlog& log, const std::string& msg);
XMLlog& operator<<(XMLlog& log, char ch);
XMLlog& operator<<(XMLlog& log, const char* str);
XMLlog& operator<<(XMLlog& log, StreamPointer func);
XMLlog& operator<<(XMLlog& log, double val);
XMLlog& operator<<(XMLlog& log, float val);
XMLlog& operator<<(XMLlog& log, int val);
XMLlog& operator<<(XMLlog& log, size_t val);
XMLlog& operator<<(XMLlog& log, const std::unordered_map<std::string, double>& dic);
XMLlog& operator<<(XMLlog& log, const std::list<std::pair<std::string, double> >& dic);

struct XMLlog {
public:
  XMLlog();
  XMLlog(std::ostream& stream);
  XMLlog(std::string filename);
  ~XMLlog();
  void begin(std::string node);
  void log(const std::string& msg);

  /* log an attribute such as <entry key=" " val = " "> */
  template<class T>
  void logAttr(const std::string& name, const std::string& key, const T val) {
    this->logRaw("<" + name + " name=\"");
    this->log(key);
    this->logRaw("\" value=\"");
    this->log(std::to_string(val));
    this->logRaw("\"/>");
    (*this) << std::endl; 
  }

  void logRaw(const std::string& msg);
  void end();
  static std::string encodeString(std::string source);
  static const char endl = '\n';
  size_t depth() const {return stack.size(); }

  std::ostream* stream;
private:
  std::mutex th_mutex;
  std::unique_lock<std::mutex> th_lock;
  std::vector<std::string> stack;  
  bool lets_close;
  void registerSignals();
};


#endif
