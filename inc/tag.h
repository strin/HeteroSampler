#ifndef POS_TAG_H
#define POS_TAG_H

#include <string>
#include "corpus.h"
#include "objcokus.h"

#include <map>
#include <memory>
#include <boost/random/uniform_int.hpp>

typedef std::shared_ptr<std::map<std::string, int> > FeaturePointer;
typedef std::shared_ptr<std::map<std::string, double> > ParamPointer;
static ParamPointer makeParamPointer() {
  return ParamPointer(new std::map<std::string, double>());
}
static FeaturePointer makeFeaturePointer() {
  return FeaturePointer(new std::map<std::string, int>());
}

static std::string str(FeaturePointer features);

struct Tag {
public:
  const Sentence* seq;
  const Corpus& corpus;
  
  objcokus* rng;

  std::vector<int> tag;

  FeaturePointer features; 
  ParamPointer param;

  /* corpus should be training corpus, as its tag mapping would be used.
   * DO NOT use the test corpus, as it would confuse the tagging.
   */
  Tag(const Sentence* seq, const Corpus& corpus, 
     objcokus* rng, ParamPointer param);
  size_t size() const {return this->tag.size(); }
  void randomInit();
  ParamPointer proposeGibbs(int pos, bool withgrad = false);
  FeaturePointer extractFeatures(const std::vector<int>& tag);
  double score(FeaturePointer features); // return un-normalized log-score.
  std::string str(); 
};

#endif
