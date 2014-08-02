#ifndef POS_MARKOV_TREE
#define POS_MARKOV_TREE

#include "tag.h"
#include "utils.h"

/* represent A Markov Transition. */
struct MarkovTreeNode {
public:
  MarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent);
  bool is_split();
  /* weighting convention 
     gradient: sum of weights of node and descendants. 
     posgrad : weight of node. 
     neggrad : sum weights of descendants */
  ParamPointer gradient, posgrad, neggrad;
  std::shared_ptr<Tag> tag; // tag after the transition.
  double log_weight;        // posterior weight for gradient.
  double log_prior_weight;  // prior weight from proposal.
  int depth;
  std::weak_ptr<MarkovTreeNode> parent; // weak_ptr: avoid cycle in reference count.
  std::vector<std::shared_ptr<MarkovTreeNode> > children;
  FeaturePointer stop_feat;            
};

static std::shared_ptr<MarkovTreeNode> makeMarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent) {
  return std::shared_ptr<MarkovTreeNode>(new MarkovTreeNode(parent));
}

typedef std::pair<std::vector<FeaturePointer>, std::vector<double> > StopDataset;
typedef std::shared_ptr<StopDataset> StopDatasetPtr;
inline static StopDatasetPtr makeStopDataset() {
  return StopDatasetPtr(new StopDataset());
}
typedef std::shared_ptr<MarkovTreeNode> MarkovTreeNodePtr;
inline static void incrStopDataset(StopDatasetPtr data, FeaturePointer stop_feat, double val) {
  data->first.push_back(stop_feat);
  data->second.push_back(val);
}
inline static void mergeStopDataset(StopDatasetPtr to, StopDatasetPtr from) {
  for(size_t i = 0; i < from->first.size(); i++) {
    to->first.push_back(from->first[i]);
    to->second.push_back(from->second[i]);
  }
}
inline static void logStopDataset(StopDatasetPtr data, XMLlog& log) {
  for(size_t i = 0; i < data->first.size(); i++) {
    log.begin("data");
    log << *data->first[i];
    log.begin("value"); log << data->second[i] << std::endl; log.end();
    log.end();
  }
}
struct MarkovTree {
public:
  MarkovTree();
  std::shared_ptr<MarkovTreeNode> root; 

  // return log(sum(posterior weights of all nodes)).
  double logSumWeights(MarkovTreeNodePtr node); 
  // return log(sum(prior weights of all nodes)).
  double logSumPriorWeights(MarkovTreeNodePtr node);
  // return log expected reward starting from a ndoe (include).
  double aggregateReward(MarkovTreeNodePtr node, double normalize);
  // generate stop dataset from split nodes.
  StopDatasetPtr generateStopDataset(MarkovTreeNodePtr node);
  // return expected value of the gradient (unnormalized).
  std::pair<ParamPointer,double> aggregateGradient(std::shared_ptr<MarkovTreeNode> node, double normalize);
  // return expected gradient.
  ParamPointer expectedGradient();
  // return all samples.
  TagVector getSamples();
  TagVector getSamples(std::shared_ptr<MarkovTreeNode> node);
};

#endif
