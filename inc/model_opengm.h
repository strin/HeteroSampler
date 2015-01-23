#pragma once

#include "utils.h"
#include "model.h"
#include "opengm.h"
#include "corpus_opengm.h"

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

    virtual void sampleOne(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature = true);

    virtual double score(const GraphicalModel& gm);

    virtual TagVector sample(const Instance& seq, bool argmax = false) {
      throw "not implemented.";
    }

    // create a new sample from an instance.
    virtual ptr<GraphicalModel> makeSample(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const;

    // create a new sample with value equal to ground truth.
    virtual ptr<GraphicalModel> makeTruth(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const;

    // create a new sample by copying an old one.
    virtual ptr<GraphicalModel> copySample(const GraphicalModel& gm) const;


    // return the Markov blanket of the node.
    // default: return the Markov blanket of node *id*
    virtual vec<int> markovBlanket(const GraphicalModel& gm, int pos) {
      auto inst = dynamic_cast<const InstanceOpenGM<GM>* >(gm.seq);
      return inst->markovBlanket(pos);
    }

    // return the nodes whose Markov blanket include the node. 
    // default: return the Markov blanket of node *id*
    virtual vec<int> invMarkovBlanket(const GraphicalModel& gm, int pos) {
      auto inst = dynamic_cast<const InstanceOpenGM<GM>* >(gm.seq);
      return inst->invMarkovBlanket(pos);
    }

    string annealing;
    double temp, temp_decay, temp_magnify, temp_init;
  };

  template<class GM, class ACC>
  ModelEnumerativeGibbs<GM, ACC>::ModelEnumerativeGibbs(const boost::program_options::variables_map& vm)
  : Model(nullptr, vm) , 
    annealing(vm["temp"].as<string>()) {
    if(annealing == "scanline") { // use the annealing scheme introduced in scanline paper (CVPR 2014).
      temp_decay = vm["temp_decay"].as<double>();
      temp_magnify = vm["temp_magnify"].as<double>();
    }
    temp_init = vm["temp_init"].as<double>();
  }

  template<class GM, class ACC> 
  double ModelEnumerativeGibbs<GM, ACC>::score(const GraphicalModel& gm) {
    auto& opengm_ = dynamic_cast<const OpenGM<GraphicalModelType>& >(gm);
    double score = opengm_.gm_.evaluate(opengm_.getLabels());
    if(typeid(AccumulationType) == typeid(opengm::Maximizer)) { // Maximum probability.
      score = log(score);
    }else if(typeid(AccumulationType) == typeid(opengm::Minimizer)) { // Minimize energe.
      score = -score;
    }
    return score;
  }

  template<class GM, class ACC> 
  ptr<GraphicalModel> ModelEnumerativeGibbs<GM, ACC>::makeSample(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const {
    auto instance_opengm = dynamic_cast<const InstanceOpenGM<GM>& >(instance);
    return std::make_shared<OpenGM<GM> >(&instance, *instance_opengm.gm);
  }

  template<class GM, class ACC>
  ptr<GraphicalModel> ModelEnumerativeGibbs<GM, ACC>::makeTruth(const Instance& instance, ptr<Corpus> corpus, objcokus* rng) const {
    throw "no truth available in opengm.";
  }

  template<class GM, class ACC>
  ptr<GraphicalModel> ModelEnumerativeGibbs<GM, ACC>::copySample(const GraphicalModel& gm) const {
    auto opengm_ = dynamic_cast<const OpenGM<GraphicalModelType>& >(gm);
    return std::make_shared<OpenGM<GraphicalModelType> >(opengm_);
  }



  template<class GM, class ACC>
  void ModelEnumerativeGibbs<GM, ACC>::sampleOne(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature) {
    if(choice >= (int)gm.size()) 
      throw "Gibbs sampling proposal out of bound.";
    auto& opengm_ = dynamic_cast<OpenGM<GraphicalModelType>& >(gm);
    vec<double> sc(gm.numLabels(choice));
    auto computeSc = [&] (int choice) {
      for(size_t t = 0; t < gm.numLabels(choice); t++) {
        ValueType value = opengm_.valueAfterMove(&choice, &choice + 1, &t);
        double score = (double)value;
        if(typeid(AccumulationType) == typeid(opengm::Maximizer)) { // Maximum probability.
          score = log(score);
        }else if(typeid(AccumulationType) == typeid(opengm::Minimizer)) { // Minimize energe.
          score = -score;
        }
        sc[t] = score / temp;
      }
      logNormalize(&sc[0], gm.numLabels(choice));
    };
    /* estimate temperature */
    if(gm.time == 0) {    // compute initial temperature.
      temp = temp_init;
      if(annealing == "scanline") {
        double q = 0;
        vec<LabelType> labels = opengm_.getLabels();
        for(int i = 0; i < (int)gm.size(); i++) {
          computeSc(i);
          q -= sc[labels[i]];
        }
        q /= (double)gm.size();
        temp = temp_magnify * q;
      }
    }else if(gm.time % gm.size() == 0 and use_meta_feature) {
      if(annealing == "scanline") {
        temp = temp * temp_decay;
      }
    }

    /* sampling */
    computeSc(choice);
    size_t val = rng.sampleCategorical(&sc[0], gm.numLabels(choice));
    size_t oldval = opengm_.state(choice);
    if(use_meta_feature) {
      gm.oldlabels[choice] = oldval;
    }
    gm.reward[choice] = (sc[val] - sc[oldval]) * temp;  // use reward without temperature.
    gm.sc = sc;
    
    if(use_meta_feature) {
      gm.prev_sc[choice] = gm.this_sc[choice];
      gm.this_sc[choice] = sc;
      gm.prev_entropy[choice] = gm.entropy[choice];
      gm.entropy[choice] = logEntropy(&sc[0], gm.numLabels(choice));
      gm.timestamp[choice]++;
      gm.time++;
    }

    /* compute stats */
    opengm_.move(&choice, &choice + 1, &val);
  }

}