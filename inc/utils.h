#ifndef POS_UTIL_H
#define POS_UTIL_H

#include <cmath>
#include <map>
#include "float.h"

static void logNormalize(double* logprob, int len) {
  double lse = -DBL_MAX;
  for(int i = 0; i < len; i++) {
    if(lse == -DBL_MAX) lse = logprob[i];
    else if(lse < logprob[i]) lse = logprob[i]+log(1+exp(lse-logprob[i]));
    else lse = lse+log(1+exp(logprob[i]-lse));
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
