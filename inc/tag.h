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
  inline size_t size() const {return this->tag.size(); }

  // overide: number of possible tags.
  inline size_t numLabels(int id) const {return this->corpus->tags.size();}

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

  // get tag of a node.
  virtual int getLabel(int id) const {
    assert(id >= 0 and id < this->size());
    return tag[id];
  }

  // set tag of a node.
  virtual void setLabel(int id, int val) {
    tag[id] = val;
  }

  // return the string of a tag.
  std::string getTag(size_t pos) const;

  // to string. 
  virtual std::string str(bool verbose = false); 
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

