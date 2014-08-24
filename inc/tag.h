#ifndef POS_TAG_H
#define POS_TAG_H

#include <string>
#include "corpus.h"
#include "objcokus.h"

#include <map>
#include <list>
#include <unordered_map>
#include <memory>
#include <boost/random/uniform_int.hpp>

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

inline static void insertFeature(FeaturePointer featA, FeaturePointer featB) {
  featA->insert(featA->end(), featB->begin(), featB->end());
}

inline static Vector2d makeVector2d(size_t m, size_t n, double c = 0.0) {
  Vector2d vec(m);
  for(size_t mi = 0; mi < m; mi++) vec[mi].resize(n, c);
  return vec;
}

inline static Vector2d operator+(const Vector2d& a, const Vector2d& b) {
  assert(a.size() == b.size() and a.size() > 0);
  Vector2d c(a.size());
  for(size_t i = 0; i < a.size(); i++) {
    assert(a[i].size() == b[i].size());
    c[i].resize(a[i].size());
    for(size_t j = 0; j < a[i].size(); j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  return c;
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

inline static std::string str(FeaturePointer features);

struct Tag {
public:
  const Sentence* seq;
  const Corpus* corpus;
  
  objcokus* rng;

  std::vector<int> tag;
  std::vector<double> entropy; // current entropy when sampled.
  std::vector<double> sc;      // max score (normalized) when sampled.
  std::vector<double> resp;
  std::vector<int> mask;

  FeaturePointer features; 
  ParamPointer param;

  /* corpus should be training corpus, as its tag mapping would be used.
   * DO NOT use the test corpus, as it would confuse the tagging.
   */
  Tag(const Sentence* seq, const Corpus* corpus, 
     objcokus* rng, ParamPointer param); // random init tag.
  Tag(const Sentence& seq, const Corpus* corpus, 
     objcokus* rng, ParamPointer param); // copy tag from seq.
  // length of sequence.
  inline size_t size() const {return this->tag.size(); }
  // initialize the sequence tags uniformly at random.
  void randomInit();
  // propose Gibbs-style modification to *pos*
  // return: gradient induced by Gibbs kernel.
  ParamPointer proposeGibbs(int pos, std::function<FeaturePointer(const Tag& tag)> featExtract, bool grad_expect =  false, bool grad_sample = true, bool argmax = false);
   // return un-normalized log-score.
  double score(FeaturePointer features) const; 
  // distance to another tag.
  // warning: both tags should have same length and dict. 
  double distance(const Tag& tag);  
  // return the string of a tag.
  std::string getTag(size_t pos) const;
  // to string. 
  std::string str(); 
};

typedef std::shared_ptr<Tag> TagPtr;
typedef std::vector<TagPtr> TagVector;
inline static TagPtr makeTagPtr(const Tag& tag) {
  return std::shared_ptr<Tag>(new Tag(tag)); 
}
inline static TagPtr makeTagPtr(const Sentence* seq, const Corpus* corpus, objcokus* rng, ParamPointer param) {
  return std::shared_ptr<Tag>(new Tag(seq, corpus, rng, param));
}


#endif
