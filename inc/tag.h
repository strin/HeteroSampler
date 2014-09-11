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

namespace Tagging {
  inline static std::string str(FeaturePointer features);

  struct Tag {
  public:
    const Sentence* seq;
    ptr<Corpus> corpus;
    
    objcokus* rng;

    std::vector<int> tag;
    vec<double> timestamp;                  // whenever a position is changed, its timestamp is incremented.
    vec<double> checksum;                   // if checksum is changes, then the position might be updated.
    std::vector<double> entropy;            // current entropy when sampled.
    std::vector<double> sc;
    std::vector<double> entropy_unigram;    // unigram entropy of positions.
    vec<vec<double> > sc_unigram;

    std::vector<double> reward;
    std::vector<double> resp;
    std::vector<int> mask;
    std::vector<FeaturePointer> feat;

    FeaturePointer features; 
    ParamPointer param;

    /* corpus should be training corpus, as its tag mapping would be used.
     * DO NOT use the test corpus, as it would confuse the tagging.
     */
    Tag(const Sentence* seq, ptr<Corpus> corpus, 
       objcokus* rng, ParamPointer param); // random init tag.
    Tag(const Sentence& seq, ptr<Corpus> corpus, 
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
  inline static TagPtr makeTagPtr(const Sentence* seq, ptr<Corpus> corpus, objcokus* rng, ParamPointer param) {
    return std::shared_ptr<Tag>(new Tag(seq, corpus, rng, param));
  }

}
#endif
