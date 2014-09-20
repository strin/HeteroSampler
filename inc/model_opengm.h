#pragma once

#include "utils.h"
#include "model.h"
#include "opengm.h"

namespace Tagging {
  // OpenGM inference algorithms invoke existing trained graphical model. 
  // so all gradients in Model is 0.
  template<class GM, class ACC>
  struct ModelEnumerativeGibbs : public Model {
    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;
    OPENGM_GM_TYPE_TYPEDEFS;
    ModelEnumerativeGibbs(const boost::program_options::variables_map& vm);

    /* implement interface in model */
    virtual ParamPointer gradient(const Instance& seq) {
      throw "OpenGM does not support gradient.";
    }

    virtual ParamPointer sampleOne(GraphicalModel& gm, objcokus& rng, int choice);

    virtual double score(const GraphicalModel& gm);

    virtual TagVector sample(const Instance& seq, bool argmax = false) {
      throw "not implemented.";
    }
  };

  template<class GM, class ACC>
  ModelEnumerativeGibbs<GM, ACC>::ModelEnumerativeGibbs(const boost::program_options::variables_map& vm)
  : Model(nullptr, vm) {

  }

  template<class GM, class ACC> 
  double ModelEnumerativeGibbs<GM, ACC>::score(const GraphicalModel& gm) {
    auto& opengm_ = dynamic_cast<const OpenGM<GraphicalModelType>& >(gm);
    return opengm_.gm_.evaluate(opengm_.getLabels());
  }

  template<class GM, class ACC>
  ParamPointer ModelEnumerativeGibbs<GM, ACC>::sampleOne(GraphicalModel& gm, objcokus& rng, int choice) {
    if(choice >= (int)gm.size()) 
      throw "Gibbs sampling proposal out of bound.";
    auto& opengm_ = dynamic_cast<OpenGM<GraphicalModelType>& >(gm);
    gm.sc.clear();
    for(size_t t = 0; t < gm.numLabels(choice); t++) {
      ValueType value = opengm_.valueAfterMove(&choice, &choice + 1, &t);
      double score = (double)value;
      if(typeid(AccumulationType) == typeid(opengm::Maximizer)) { // Maximum probability.
        score = log(score);
      }else if(typeid(AccumulationType) == typeid(opengm::Minimizer)) { // Minimize energe.
        score = -score;
      }
      gm.sc.push_back(score);
    }
    logNormalize(&gm.sc[0], gm.numLabels(choice));
    size_t val = rng.sampleCategorical(&gm.sc[0], gm.numLabels(choice));
    opengm_.move(&choice, &choice + 1, &val);
    return nullptr;
  }

}