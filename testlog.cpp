#include "log.h"
#include <iostream>

using namespace std;

int main() {
  XMLlog log;
  log.begin("track");
  log << "that's cute." << endl;
  log.begin("iteration_1");
  log << "hello man" << endl;
  log.end();
  log.end();
  return 0;
}
