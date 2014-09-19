#pragma once
#include "utils.h"
#include "corpus.h"
#include "gm.h"

namespace Tagging {
// output feature vector of an instance as std::string.
inline static std::string str(FeaturePointer features);

struct Tag : public GraphicalModel {
public:
  std::vector<int> tag;

  FeaturePointer features; 
  ParamPointer param;

  /* corpus should be training corpus, as its tag mapping would be used.
   * DO NOT use the test corpus, as it would confuse the tagging.
   */
  Tag(const Instance* seq, ptr<Corpus> corpus, 
     objcokus* rng, ParamPointer param); // random init tag.
  Tag(const Instance& seq, ptr<Corpus> corpus, 
     objcokus* rng, ParamPointer param); // copy tag from seq.
  // length of sequence.
  size_t size() const {return this->tag.size(); }
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

inline static TagPtr makeTagPtr(const Instance* seq, ptr<Corpus> corpus, objcokus* rng, ParamPointer param) {
  return std::shared_ptr<Tag>(new Tag(seq, corpus, rng, param));
}

}

