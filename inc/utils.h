#ifndef POS_UTIL_H
#define POS_UTIL_H

#include <cmath>
#include <map>
#include <string>
#include <iostream>
#include "float.h"
#include <functional>
#include <algorithm>
#include <sstream>

inline static double logisticFunc(double z) {
  return 1./(1+exp(0-z));
}

static long getFingerPrint(long iterations, long startSeed) { // random hash function taken from 6.816.
  const long m = (long) 0xFFFFFFFFFFFFL;
  const long a = 25214903917L;
  const long c = 11L;
  long seed = startSeed;
  for(long i = 0; i < iterations; i++) {
    seed = (seed*a + c) & m;
  }
  return ( seed >> 12 );
}

static double logAdd(double a, double b) { // add in log space.
  if(a == -DBL_MAX) return b;
  if(b == -DBL_MAX) return a;
  if(a < b) return b+log(1+exp(a-b)); // may overflow, ignore for now.
  else return a+log(1+exp(b-a));
}

static void logNormalize(double* logprob, int len) {
  double lse = -DBL_MAX;
  for(int i = 0; i < len; i++) {
    lse = logAdd(lse, logprob[i]); 
  }
  for(int i = 0; i < len; i++) {
    logprob[i] -= lse;
  }
}

template<class K, class T>
static void mapUpdate(std::map<std::string, K>& g, const std::map<std::string, T>& u, double eta = 1.0) {
  for(const std::pair<std::string, T>& p : u) {
    if(g.find(p.first) == g.end())
      g[p.first] = (K)0.0;
    g[p.first] += (K)p.second * eta;
  }
}

template<class K>
static void mapUpdate(std::map<std::string, K>& g, std::string key, K val) {
  if(g.find(key) == g.end())
    g[key] = (K)0.0;
  g[key] += val;
}

template<class K>
static void mapDivide(std::map<std::string, K>& g, double eta) {
  for(const std::pair<std::string, K>& p : g) {
    g[p.first] = p.second/(K)eta;
  }
}


#endif
