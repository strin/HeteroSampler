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

typedef std::list<FeaturePointer> StopDatasetKeyContainer;
typedef std::list<double> StopDatasetValueContainer;
typedef std::list<Tag> StopDatasetSeqContainer;
typedef std::tuple<StopDatasetKeyContainer, StopDatasetValueContainer, StopDatasetSeqContainer> StopDataset;
typedef std::shared_ptr<StopDataset> StopDatasetPtr;
inline static StopDatasetPtr makeStopDataset() {
  return StopDatasetPtr(new StopDataset());
}
typedef std::shared_ptr<MarkovTreeNode> MarkovTreeNodePtr;
inline static void incrStopDataset(StopDatasetPtr data, FeaturePointer stop_feat, double val, Tag seq) {
  std::get<0>(*data).push_back(stop_feat);
  std::get<1>(*data).push_back(val);
  std::get<2>(*data).push_back(seq);
}

inline static void mergeStopDataset(StopDatasetPtr to, StopDatasetPtr from) {
  StopDatasetKeyContainer::iterator key_iter;
  StopDatasetValueContainer::iterator value_iter;
  StopDatasetSeqContainer::iterator seq_iter;
  for(key_iter = std::get<0>(*from).begin(), value_iter = std::get<1>(*from).begin(),
      seq_iter = std::get<2>(*from).begin();
      key_iter != std::get<0>(*from).end() && value_iter != std::get<1>(*from).end(),
      seq_iter != std::get<2>(*from).end(); 
      key_iter++, value_iter++, seq_iter++) {
    std::get<0>(*to).push_back(*key_iter);
    std::get<1>(*to).push_back(*value_iter);
    std::get<2>(*to).push_back(*seq_iter);
  }
}

inline static void truncateStopDataset(StopDatasetPtr dataset, size_t size) {
  while(std::get<0>(*dataset).size() > size) {
    std::get<0>(*dataset).pop_front();
    std::get<1>(*dataset).pop_front();
    std::get<2>(*dataset).pop_front();
  }
}

inline static void logStopDataset(StopDatasetPtr data, XMLlog& log) {
  StopDatasetKeyContainer::iterator key_iter;
  StopDatasetValueContainer::iterator value_iter;
  StopDatasetSeqContainer::iterator seq_iter;
  for(key_iter = std::get<0>(*data).begin(), value_iter = std::get<1>(*data).begin(),
      seq_iter = std::get<2>(*data).begin();
      key_iter != std::get<0>(*data).end() && value_iter != std::get<1>(*data).end(),
      seq_iter != std::get<2>(*data).end(); 
      key_iter++, value_iter++, seq_iter++) {
    log.begin("data");
    log << **key_iter;
    log.begin("value"); log << *value_iter << std::endl; log.end();
    log.begin("truth"); 
    log << seq_iter->seq->str() << std::endl;
    log.end();
    log.begin("tag");
    log << seq_iter->str() << std::endl;
    log.end();
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
