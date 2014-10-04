#ifndef TAGGING_UTILS
#define TAGGING_UTILS

#include <cmath>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <set>
#include <list>
#include <iostream>
#include <functional>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <ctime>
#include <memory>
#include <thread>
#include <condition_variable>

#include <boost/algorithm/string.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/program_options.hpp>
#include <boost/heap/fibonacci_heap.hpp>

#include "dirent.h"

#include "float.h"
#include "log.h"
#include "stdlib.h"
#include "objcokus.h"

namespace Tagging {
  template<class T>
  using ptr = std::shared_ptr<T>;

  using string = std::string;

  using std::cout;
  using std::cin;
  using std::cerr;
  using std::endl;
  using std::to_string;

  template<class T>
  using vec = std::vector<T>;

  template<class T>
  using set = std::set<T>;

  template<class T>
  using vec2d = vec<vec<T> >;

  template<class T>
  using ptrs = vec<ptr<T> >;

  template<class T, class... Args>
  ptr<T> make_shared(Args&&... args) {
    return ptr<T>(new T(std::forward<Args>(args)...));
  }

  template<class K, class T>
  class map : public std::unordered_map<K, T> {
  public:
    bool contains(K key) {
      return this->find(key) != this->end();
    }
  };

  template<class T>
  using list = std::list<T>;

  template<class A, class B>
  using pair = std::pair<A, B>;

  template<class... Ts>
  using tuple = std::tuple<Ts...>;

  typedef std::pair<std::string, double> ParamItem;
  typedef ParamItem FeatureItem;
  typedef std::shared_ptr<std::unordered_map<std::string, double> > ParamPointer;
  // typedef ParamPointer FeaturePointer;
  typedef std::shared_ptr<std::list<std::pair<std::string, double> > > FeaturePointer;
  typedef std::vector<std::vector<double> > Vector2d;

  inline static ParamPointer makeParamPointer() {
    return ParamPointer(new std::unordered_map<std::string, double>());
  }

  inline static FeaturePointer makeFeaturePointer() {
    // return makeParamPointer();
    return FeaturePointer(new std::list<std::pair<std::string, double> >());
  }

  inline static void insertFeature(FeaturePointer feat, const std::string& key, double val = 1.0) {
    feat->push_back(std::make_pair(key, val));
  }

  inline static double* findFeature(FeaturePointer feat, const std::string& key) {
    for(auto& pair : *feat) {
      if(pair.first == key) {
        return &pair.second;
      }
    }
    return nullptr;
  }

  inline static double getFeature(FeaturePointer feat, const std::string& key) {
    for(auto& pair : *feat) {
      if(pair.first == key) {
        return pair.second;
      }
    }
    return 0;
  }

  inline static void insertFeature(FeaturePointer featA, FeaturePointer featB) {
    featA->insert(featA->end(), featB->begin(), featB->end());
  }

  inline static Vector2d makeVector2d(size_t m, size_t n, double c = 0.0) {
    Vector2d vec(m);
    for(size_t mi = 0; mi < m; mi++) vec[mi].resize(n, c);
    return vec;
  }

  inline static double score(ParamPointer param, FeaturePointer feat) {
    double ret = 0.0;
    for(const std::pair<std::string, double>& pair : *feat) {
      if(param->find(pair.first) != param->end()) {
	ret += (*param)[pair.first] * pair.second;
      }
    }
    return ret;
  }

  inline static void copyParamFeatures(ParamPointer param_from, std::string prefix_from,
				  ParamPointer param_to, std::string prefix_to) {
    for(const std::pair<std::string, double>& pair : *param_from) {
      std::string key = pair.first;
      size_t pos = key.find(prefix_from);
      if(pos == std::string::npos) 
	continue;
      (*param_to)[prefix_to + key.substr(pos + prefix_from.length())] = pair.second;
    }
  }

  inline static double logisticFunc(double z) {
    return 1./(1+exp(0-z));
  }

  template<class T, class K>
  static inline bool isinstance(K t) {
    return std::dynamic_pointer_cast<T>(t) != nullptr;
  }

  template<class T, class K>
  static inline ptr<T> cast(K t) {
    assert(isinstance<T>(t));
    return std::dynamic_pointer_cast<T>(t);
  }

  template<class T, class K>
  static inline ptrs<T> castVector(ptrs<K> t) {
    ptrs<T> ret;
    for(auto p : t) {
      assert(isinstance<T>(p));
      ret.push_back(cast<T>(p));
    }
    return ret;
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
  static void mapReset(std::unordered_map<std::string, K>& g, double eta = 1.0) {
    for(const std::pair<std::string, K>& p : g) {
      g[p.first] = eta;
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

  // obtain git hash of the current commit under current directory.
  static string getGitHash() {
    FILE* pipe = popen("git rev-parse HEAD", "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
      if(fgets(buffer, 128, pipe) != NULL)
	result += buffer;
    }
    pclose(pipe);
    return result;
  }
}

#endif
