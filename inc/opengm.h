#pragma once

#include "utils.h"
#include "gm.h"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/inference.hxx"

namespace Tagging {


template<class GM>
struct OpenGM 
: public GraphicalModel, public opengm::Movemaker<GM> {
public:
  typedef opengm::Movemaker<GM> MovemakerType;
  // using typename MovemakerType::GraphicalModelType;
  typedef GM GraphicalModelType;
  OPENGM_GM_TYPE_TYPEDEFS;

  OpenGM(const Instance* seq, const GraphicalModelType& gm);

  vec<LabelType> getLabels() const;

  /* implement GraphicalModel interface */
  virtual string str();
  virtual size_t size() const;
  virtual size_t numLabels(int id) const;

  using MovemakerType::gm_;
};

template<class GM>
OpenGM<GM>::OpenGM(const Instance* seq, const GraphicalModelType& gm)
: MovemakerType(gm) {
  this->seq = seq;
  size_t size = gm.numberOfVariables();
  timestamp.resize(size, 0);
  checksum.resize(size);
  reward.resize(size, 0);
  entropy.resize(size);
  entropy_unigram.resize(size);
  resp.resize(size, 0);
  mask.resize(size, 0);
  feat.resize(size);
}

template<class GM>
vec<typename OpenGM<GM>::LabelType> OpenGM<GM>::getLabels() const { // why cannot use vec<LabelType>.
  vec<LabelType> ret(gm_.numberOfVariables());
  for(size_t j = 0; j < ret.size(); ++j) {
     // if(!inInference_)
     //    x[j] = currentBestState_[j];
    ret[j] = opengm::Movemaker<GM>::state(j);
  }
  return ret;
}

template<class GM>
string OpenGM<GM>::str() {
  return "not available";
}

template<class GM>
size_t OpenGM<GM>::size() const {
  return gm_.numberOfVariables();
}

template<class GM>
size_t OpenGM<GM>::numLabels(int id) const {
  return gm_.numberOfLabels(id);
}


}