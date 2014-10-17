// Learning inference policies that deal with entire dataset. 
//
#ifndef POS_BLOCK_POLICY_H
#define POS_BLOCK_POLICY_H

#include "utils.h"
#include "model.h"
#include "MarkovTree.h"
#include "policy.h"
#include "tag.h"
#include "float.h"
#include "ThreadPool.h"
#include <boost/program_options.hpp>

namespace Tagging {

template<class PolicyType>
class BlockPolicy : public PolicyType {
public:
  BlockPolicy(ModelPtr model, const boost::program_options::variables_map& vm)
  : PolicyType(model, vm) {
  }

  ~BlockPolicy() {
  }

  class Result : public PolicyType::Result {
  public:
    Result(ptr<Corpus> corpus)
    : PolicyType::Result(corpus) {
    
    }
    Heap heap;
  };

  using PolicyType::lg;
  using PolicyType::Q;
  using PolicyType::param;
  using Policy::featopt;
  using Policy::featoptFind;
  
  virtual void train(ptr<Corpus> corpus) {
    lg->begin("train");
      /* train the model */
    if(Q > 0)
      cout << "start training ... " << endl;
    for(size_t q = 0; q < Q; q++) {
      cout << "\t epoch " << q << endl;
      cout << "\t update policy " <<  endl;
      PolicyType::trainPolicy(corpus);
    }
    if(Q == 0) { // notraining is needed.
      // set all feature weights to 1.
      for(const string& opt : featopt) {
        (*param)[opt] = 1;
      }
      // overwrite.
      if(featoptFind("sp"))
	(*param)["sp"] = -0.3;
    }
    /* log policy examples */
    lg->begin("policy_example");
      for(auto example : this->examples) {
        example.serialize(lg);
      }
    lg->end(); // </policy_example>
    lg->end(); // </train>
  }

  ptr<Result> test(ptr<Corpus> corpus, double budget);
  
  void test(ptr<Result> result, double budget); 

  virtual void testPolicy(ptr<Result> result, double budget);

  MarkovTreeNodePtr sampleOne(ptr<Result> result, objcokus& rng, const Location& loc);

  /* Given a forest of MarkovTreeNode 
   * return which node and which position of the node to sample.
   * */
  virtual Location policy(ptr<Result> result);

private:
  objcokus rng;
};


template<class PolicyType>
ptr<typename BlockPolicy<PolicyType>::Result>
BlockPolicy<PolicyType>::test(ptr<Corpus> corpus, double budget) {
  Policy::model->time = 0;
  auto result = std::make_shared<BlockPolicy::Result>(corpus);
  result->corpus->retag(PolicyType::model->corpus);
  result->nodes.resize(fmin((size_t)PolicyType::test_count, (size_t)corpus->seqs.size()), nullptr);
  for(size_t i = 0; i < result->size(); i++) {
    auto node = makeMarkovTreeNode(nullptr);
    node->model = PolicyType::model;
    node->gm = PolicyType::model->makeSample(*corpus->seqs[i], PolicyType::model->corpus, &rng);
    node->log_prior_weight = PolicyType::model->score(*node->gm);
    result->nodes[i] = node;
    for(int t = 0; t < node->gm->size(); t++) {
      node->gm->resp[t] = 1e8 - t; // this is a hack.
      Heap::handle_type handle = result->heap.push(Value(Location(i, t), node->gm->resp[t]));
      node->gm->handle[t] = handle;
    }
  }
  result->time = 0;
  result->wallclock = 0;
  test(result, budget);
  return result;
}

template<class PolicyType>
void 
BlockPolicy<PolicyType>::test(ptr<BlockPolicy<PolicyType>::Result> result, double budget) {
  std::cout << "> test " << std::endl;
  PolicyType::lg->begin("test");
  PolicyType::lg->begin("param");
  *PolicyType::lg << *PolicyType::param;
  PolicyType::lg->end(); // </param>
  if(budget > 0) {
    this->testPolicy(result, budget);
  }
  PolicyType::lg->end(); // </test>
}

template<class PolicyType>
void 
BlockPolicy<PolicyType>::testPolicy(ptr<BlockPolicy<PolicyType>::Result> result, double budget) {
  clock_t time_start = clock(), time_end;
  assert(result != nullptr);
  double total_budget = result->corpus->count(PolicyType::test_count) * budget;
  for(size_t b = 0; b < total_budget; b++) {
    auto p = policy(result);
//    std::cout << p.index << " , " << p.pos << std::endl;
    result->setNode(p.index, this->sampleOne(result, this->rng, p));
  }
  auto lg = PolicyType::lg;
  double hit_count = 0, pred_count = 0, truth_count = 0;
  this->lg->begin("example");
  for(size_t i = 0; i < result->size(); i++) {
    MarkovTreeNodePtr node = result->getNode(i);
    lg->begin("example_"+std::to_string(i));
    this->logNode(node);
    while(node->children.size() > 0) node = node->children[0]; // take final sample.
    if(this->model->scoring == Model::SCORING_ACCURACY) {
      tuple<int, int> hit_pred = this->model->evalPOS(*cast<Tag>(node->max_gm));
      hit_count += std::get<0>(hit_pred);
      pred_count += std::get<1>(hit_pred);
    }else if(this->model->scoring == Model::SCORING_NER) {
      tuple<int, int, int> hit_pred_truth = this->model->evalNER(*cast<Tag>(node->max_gm));
      hit_count += std::get<0>(hit_pred_truth);
      pred_count += std::get<1>(hit_pred_truth);
      truth_count += std::get<2>(hit_pred_truth);
    }else if(this->model->scoring == Model::SCORING_LHOOD) {
      hit_count += this->model->score(*node->max_gm);
      pred_count++;
    }
    lg->end(); // </example_i>
  }
  lg->end(); // </example>
  /* log summary stats */
  time_end = clock();
  double accuracy = (double)hit_count / pred_count;
  double recall = (double)hit_count / truth_count;
  result->time += total_budget / result->size();
  result->wallclock += (double)(time_end - time_start) / CLOCKS_PER_SEC;
  lg->begin("time");
    std::cout << "time: " << result->time << std::endl;
    *lg << result->time << std::endl;
  lg->end(); // </time>
  lg->begin("wallclock");
    std::cout << "wallclock: " << result->wallclock << std::endl;
    *lg << result->wallclock << std::endl;
  lg->end(); // </wallclock>
  lg->begin("wallclock_sample");
    std::cout << "wallclock_sample: " << result->wallclock_sample << std::endl;
    *lg << result->wallclock_sample << std::endl;
  lg->end();
  lg->begin("wallclock_policy");
    std::cout << "wallclock_policy: " << result->wallclock_policy << std::endl;
    *lg << result->wallclock_policy << std::endl;
  lg->end();
  if(this->model->scoring == Model::SCORING_ACCURACY) {
    lg->begin("accuracy");
    *lg << accuracy << std::endl;
    std::cout << "acc: " << accuracy << std::endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  }else if(this->model->scoring == Model::SCORING_NER) {
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    lg->begin("accuracy");
    *lg << f1 << std::endl;
    std::cout << "f1: " << f1 << std::endl;
    lg->end(); // </accuracy>
    result->score = f1;
  }else if(this->model->scoring == Model::SCORING_LHOOD) {
    lg->begin("accuracy");
    *lg << accuracy << std::endl;
    std::cout << "lhood: " << accuracy << std::endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  }
}


template<class PolicyType>
MarkovTreeNodePtr
BlockPolicy<PolicyType>::sampleOne(ptr<BlockPolicy<PolicyType>::Result> result, objcokus& rng, const Location& loc) {
  clock_t clock_start = clock(), clock_end;
  int index = loc.index, pos = loc.pos;
  MarkovTreeNodePtr node = result->getNode(index);
  node = PolicyType::sampleOne(node, rng, pos);
  clock_end = clock();
  result->wallclock_sample += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  clock_start = clock();
  PolicyType::updateResp(node, rng, pos, &result->heap);
  clock_end = clock();
  result->wallclock_policy += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  return node;
}

template<class PolicyType>
Location
BlockPolicy<PolicyType>::policy(ptr<BlockPolicy<PolicyType>::Result> result) {
  clock_t clock_start = clock(), clock_end;
  Location loc;
  Value val = result->heap.top();
  // std::cout << "val : " << val.resp << std::endl;
  clock_end = clock();
  result->wallclock_policy += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  return val.loc;
}

}

#endif
