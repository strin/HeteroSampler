#ifndef POS_UTIL_H
#define POS_UTIL_H

#include <cmath>
#include <map>
#include <unordered_map>
#include "boost/algorithm/string.hpp"
#include <string>
#include <iostream>
#include <list>
#include "float.h"
#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>
#include "log.h"
#include "stdlib.h"
#include "objcokus.h"

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
  if(a < b) return b + log(1 + exp(a - b)); // may overflow, ignore for now.
  else return a + log(1 + exp(b - a));
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

static double logEntropy(double* logprob, int len) {
  double ent = 0.0;
  for(int i = 0; i < len; i++) {
    if(logprob[i] == -DBL_MAX) continue;
    ent -= logprob[i] * exp(logprob[i]);
  }
  return ent;
}

template<class K, class T>
static void mapUpdate(std::unordered_map<std::string, K>& g, const std::unordered_map<std::string, T>& u, double eta = 1.0) {
  for(const std::pair<std::string, T>& p : u) {
    if(g.find(p.first) == g.end())
      g[p.first] = (K)0.0;
    g[p.first] += (K)p.second * eta;
  }
}

template<class K, class T>
static void mapUpdate(std::unordered_map<std::string, K>& g, const std::list<std::pair<std::string, T> >& u, double eta = 1.0) {
  for(const std::pair<std::string, T>& p : u) {
    if(g.find(p.first) == g.end())
      g[p.first] = (K)0.0;
    g[p.first] += (K)p.second * eta;
  }
}

template<class K>
static void mapUpdate(std::unordered_map<std::string, K>& g, std::string key, K val) {
  if(g.find(key) == g.end())
    g[key] = (K)0.0;
  g[key] += val;
}

template<class K>
static void mapCopy(std::unordered_map<std::string, K>& g, const std::unordered_map<std::string, K>& u) {
  for(const std::pair<std::string, K>& p : u) {
    g[p.first] = p.second;
  }
}

template<class K>
static void mapRemove(std::unordered_map<std::string, K>& g, const std::unordered_map<std::string, K>& u) {
  for(const std::pair<std::string, K>& p : u) {
    g.erase(p.first);
  }
}


template<class K>
static void mapDivide(std::unordered_map<std::string, K>& g, double eta) {
  for(const std::pair<std::string, K>& p : g) {
    g[p.first] = p.second/(K)eta;
  }
}

template<class K>
static K mapGet(std::unordered_map<std::string, K>& g, std::string key) {
  if(g.find(key) == g.end()) return (K)0.0;
  return g[key];
}

template<class T>
static void shuffle(std::vector<T>& vec, objcokus& cokus) {
  size_t size = vec.size();
  for(size_t i = size-1; i >= 1; i--) {
    size_t j = cokus.randomMT() % (i+1);
    std::swap(vec[i], vec[j]);
  }
}

#endif
