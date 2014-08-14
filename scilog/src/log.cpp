#include "log.h"
#include <ostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <string>
#include <signal.h>
#include <unistd.h>

using namespace std;

/* bookkeep all active XML logs. */
vector<XMLlog*> GLOBAL_XML_LOGS;
const char ARR_ASCII_MAP[256][8] 
= 
{
    "&#x000;",  "&#x001;",  "&#x002;",  "&#x003;",  "&#x004;",  "&#x005;",  "&#x006;",  "&#x007;",  "&#x008;",  "&#x009;",  "&#x00A;",  "&#x00B;",  "&#x00C;",  "&#x00D;",  "&#x00E;",  "&#x00F;",
    "&#x010;",  "&#x011;",  "&#x012;",  "&#x013;",  "&#x014;",  "&#x015;",  "&#x016;",  "&#x017;",  "&#x018;",  "&#x019;",  "&#x01A;",  "&#x01B;",  "&#x01C;",  "&#x01D;",  "&#x01E;",  "&#x01F;",
    "&#x020;",  "&#x021;",  "&#x022;",  "&#x023;",  "&#x024;",  "&#x025;",  "&#x026;",  "&#x027;",  "&#x028;",  "&#x029;",  "&#x02A;",  "&#x02B;",  "&#x02C;",  "&#x02D;",  "&#x02E;",  "&#x02F;",
    "&#x030;",  "&#x031;",  "&#x032;",  "&#x033;",  "&#x034;",  "&#x035;",  "&#x036;",  "&#x037;",  "&#x038;",  "&#x039;",  "&#x03A;",  "&#x03B;",  "&#x03C;",  "&#x03D;",  "&#x03E;",  "&#x03F;",
    "&#x040;",  "&#x041;",  "&#x042;",  "&#x043;",  "&#x044;",  "&#x045;",  "&#x046;",  "&#x047;",  "&#x048;",  "&#x049;",  "&#x04A;",  "&#x04B;",  "&#x04C;",  "&#x04D;",  "&#x04E;",  "&#x04F;",  
    "&#x050;",  "&#x051;",  "&#x052;",  "&#x053;",  "&#x054;",  "&#x055;",  "&#x056;",  "&#x057;",  "&#x058;",  "&#x059;",  "&#x05A;",  "&#x05B;",  "&#x05C;",  "&#x05D;",  "&#x05E;",  "&#x05F;",  
    "&#x060;",  "&#x061;",  "&#x062;",  "&#x063;",  "&#x064;",  "&#x065;",  "&#x066;",  "&#x067;",  "&#x068;",  "&#x069;",  "&#x06A;",  "&#x06B;",  "&#x06C;",  "&#x06D;",  "&#x06E;",  "&#x06F;",
    "&#x070;",  "&#x071;",  "&#x072;",  "&#x073;",  "&#x074;",  "&#x075;",  "&#x076;",  "&#x077;",  "&#x078;",  "&#x079;",  "&#x07A;",  "&#x07B;",  "&#x07C;",  "&#x07D;",  "&#x07E;",  "&#x07F;",  
    "&#x080;",  "&#x081;",  "&#x082;",  "&#x083;",  "&#x084;",  "&#x085;",  "&#x086;",  "&#x087;",  "&#x088;",  "&#x089;",  "&#x08A;",  "&#x08B;",  "&#x08C;",  "&#x08D;",  "&#x08E;",  "&#x08F;",  
    "&#x090;",  "&#x091;",  "&#x092;",  "&#x093;",  "&#x094;",  "&#x095;",  "&#x096;",  "&#x097;",  "&#x098;",  "&#x099;",  "&#x09A;",  "&#x09B;",  "&#x09C;",  "&#x09D;",  "&#x09E;",  "&#x09F;",  
    "&#x0A0;",  "&#x0A1;",  "&#x0A2;",  "&#x0A3;",  "&#x0A4;",  "&#x0A5;",  "&#x0A6;",  "&#x0A7;",  "&#x0A8;",  "&#x0A9;",  "&#x0AA;",  "&#x0AB;",  "&#x0AC;",  "&#x0AD;",  "&#x0AE;",  "&#x0AF;",
    "&#x0B0;",  "&#x0B1;",  "&#x0B2;",  "&#x0B3;",  "&#x0B4;",  "&#x0B5;",  "&#x0B6;",  "&#x0B7;",  "&#x0B8;",  "&#x0B9;",  "&#x0BA;",  "&#x0BB;",  "&#x0BC;",  "&#x0BD;",  "&#x0BE;",  "&#x0BF;",
    "&#x0C0;",  "&#x0C1;",  "&#x0C2;",  "&#x0C3;",  "&#x0C4;",  "&#x0C5;",  "&#x0C6;",  "&#x0C7;",  "&#x0C8;",  "&#x0C9;",  "&#x0CA;",  "&#x0CB;",  "&#x0CC;",  "&#x0CD;",  "&#x0CE;",  "&#x0CF;",  
    "&#x0D0;",  "&#x0D1;",  "&#x0D2;",  "&#x0D3;",  "&#x0D4;",  "&#x0D5;",  "&#x0D6;",  "&#x0D7;",  "&#x0D8;",  "&#x0D9;",  "&#x0DA;",  "&#x0DB;",  "&#x0DC;",  "&#x0DD;",  "&#x0DE;",  "&#x0DF;",  
    "&#x0E0;",  "&#x0E1;",  "&#x0E2;",  "&#x0E3;",  "&#x0E4;",  "&#x0E5;",  "&#x0E6;",  "&#x0E7;",  "&#x0E8;",  "&#x0E9;",  "&#x0EA;",  "&#x0EB;",  "&#x0EC;",  "&#x0ED;",  "&#x0EE;",  "&#x0EF;",
    "&#x0F0;",  "&#x0F1;",  "&#x0F2;",  "&#x0F3;",  "&#x0F4;",  "&#x0F5;",  "&#x0F6;",  "&#x0F7;",  "&#x0F8;",  "&#x0F9;",  "&#x0FA;",  "&#x0FB;",  "&#x0FC;",  "&#x0FD;",  "&#x0FE;",  "&#x0FF;",
};

/* on signal stop, we need to complete the log, 
 * so the file remains a legit xml */
void sighandler(int s) {
  for(XMLlog* log : GLOBAL_XML_LOGS) {
    if(log == nullptr) continue;
    while(log->depth() > 0) 
      log->end();
  }
  exit(1);
}

XMLlog::XMLlog() 
:stream(&cout), lets_close(false) {
  this->begin("document");
  registerSignals();
}

// warning: init from stream do not add "<document>".
XMLlog::XMLlog(ostream& stream)
:stream(&stream), lets_close(false) {
  registerSignals();
}

XMLlog::XMLlog(string filename) 
:stream(new ofstream()), lets_close(true) {
  ((ofstream*)stream)->open(filename.c_str());
  this->begin("document");
  registerSignals();
}

XMLlog::~XMLlog() {
  while(this->depth() > 0) {
    this->end();
  }
  if(lets_close) { 
    ((ofstream*)this->stream)->close();
    delete stream;
  }
}

string XMLlog::encodeString(string source)
{
  string ss;
  for(int i = 0; i < source.size(); i++)
  {
    if(source[i]>>7) ss += source[i];
    else if(source[i] == '\"')
      ss += "&quot;";
    else if (((source[i]) >= 32 && (source[i]) <= 37)                        
       || ((source[i]) == 39 )
       || ((source[i]) >= 42 && (source[i]) <= 59) 
       || ((source[i]) >= 64 && (source[i]) <= 122)
       || source[i] == 9)
	  ss += source[i];
    else
	ss += ARR_ASCII_MAP[(int)source[i]];
  }
  return ss;
}

void XMLlog::registerSignals() {
   struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = sighandler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);
   GLOBAL_XML_LOGS.push_back(this);
}

void XMLlog::begin(std::string node) {
  th_mutex.lock();
  boost::replace_all(node, " ", "_");
  *stream << "<" << node << ">" << endl;
  stream->flush();
  stack.push_back(node);
  th_mutex.unlock();
}

void XMLlog::end() {
  th_mutex.lock();
  *stream << "</" << stack.back() << ">" << endl;
  stream->flush();
  stack.pop_back();
  th_mutex.unlock();
}

void XMLlog::log(const string& msg) {
  th_mutex.lock();
  *stream << XMLlog::encodeString(msg);
  stream->flush();
  th_mutex.unlock();
}

void XMLlog::logRaw(const string& msg) {
  th_mutex.lock();
  *stream << msg;
  (*stream).flush();
  th_mutex.unlock();
}

XMLlog& operator<<(XMLlog& log, const std::string& msg) {
  log.log(msg); 
  return log;
}

XMLlog& operator<<(XMLlog& log, const char* str) {
  return log << string(str);
}

XMLlog& operator<<(XMLlog& log, StreamPointer func) {
  log.stream = &(func(*log.stream));
  return log;
}

XMLlog& operator<<(XMLlog& log, double val) {
  return log << to_string(val);
}

XMLlog& operator<<(XMLlog& log, char ch) {
  stringstream ss;
  ss << ch;
  return log << ss.str();
}

XMLlog& operator<<(XMLlog& log, int val) {
  return log << to_string(val);
}

XMLlog& operator<<(XMLlog& log, float val) {
  return log << to_string(val);
}

XMLlog& operator<<(XMLlog& log, size_t val) {
  return log << to_string(val);
}

XMLlog& operator<<(XMLlog& log, const unordered_map<string, double>& dic) {
  for(const pair<string, double>& item : dic) {
    log.logRaw("<entry name=\"");
    string key = item.first;
    log.log(key);
    log.logRaw("\" value=\"");
    log.log(to_string(item.second));
    log.logRaw("\"/>");
    log << endl;
  }
  return log;
}

XMLlog& operator<<(XMLlog& log, const list<pair<string, double> >& dic) {
  for(const pair<string, double>& item : dic) {
    log.logRaw("<entry name=\"");
    string key = item.first;
    log.log(key);
    log.logRaw("\" value=\"");
    log.log(to_string(item.second));
    log.logRaw("\"/>");
    log << endl;
  }
  return log;
}
