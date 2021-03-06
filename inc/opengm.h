#pragma once

#include "utils.h"
#include "gm.h"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/inference.hxx"

namespace HeteroSampler {


template<class GM>
struct OpenGM 
: public GraphicalModel, public opengm::Movemaker<GM> {
public:
  typedef opengm::Movemaker<GM> MovemakerType;
  // using typename MovemakerType::GraphicalModelType;
  typedef GM GraphicalModelType;
  OPENGM_GM_TYPE_TYPEDEFS;

  OpenGM(const Instance* seq, const GraphicalModelType& gm);

  virtual int getLabel(int id) const {
    return (int)opengm::Movemaker<GM>::state(id);
  }

  virtual void setLabel(int id, int val) {
    this->move(&id, &id + 1, &val);
  }
  
  vec<LabelType> getLabels() const;

  /* implement GraphicalModel interface */
  virtual string str(bool verbose = false);
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
  reward.resize(size, 0);
  entropy_unigram.resize(size);
  resp.resize(size, DBL_MAX);
  mask.resize(size, 0);
  this->initStats();
}

template<class GM>
vec<typename OpenGM<GM>::LabelType> OpenGM<GM>::getLabels() const { // why cannot use vec<LabelType>.
  vec<LabelType> ret(gm_.numberOfVariables());
  for(size_t j = 0; j < ret.size(); ++j) {
    ret[j] = this->getLabel(j);
  }
  return ret;
}

template<class GM>
string OpenGM<GM>::str(bool verbose) {
  auto ret = getLabels();
  string res;
  int count = 0;
  for(auto& val : ret) {
    if(verbose) {
      res += to_string(val) + " / " + to_string(count) + "\t";
    }else{
      res += to_string(val) + "\t";
    }
    count++;
  }
  return res;
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
