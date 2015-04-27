#ifndef HETEROSAMPLER_MARKOVTREE
#define HETEROSAMPLER_MARKOVTREE

#include "tag.h"
#include "utils.h"
#include "model.h"
#include "gm.h"

namespace HeteroSampler {
  struct Model; 
  
  /* warning: this class is not thread safe */
  struct MarkovTreeNode {
  public:
    MarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent);
    bool is_split();
    bool is_leaf();
    /* weighting convention 
       gradient: sum of weights of node and descendants. 
       posgrad : weight of node. 
       neggrad : sum weights of descendants */
    ParamPointer gradient, posgrad, neggrad, G2;
    ptr<Model> model;
    std::shared_ptr<GraphicalModel> gm; // tag after the transition.
    double log_weight;        // posterior weight for gradient.

    double log_prior_weight;  // prior weight from proposal.
    double max_log_prior_weight;
    std::shared_ptr<GraphicalModel> max_gm;  // save gm with maximum score.


    int depth;                // how many samples have been generated.
    Location choice;               // if using a policy, which choice is made?
    size_t time_stamp;        // time stamp of this object.
    std::weak_ptr<MarkovTreeNode> parent; // weak_ptr: avoid cycle in reference count.
    std::vector<std::shared_ptr<MarkovTreeNode> > children;
    FeaturePointer stop_feat;            
    bool compute_stop;
    double resp;             // response for stop or not prediction.
  };

  typedef std::shared_ptr<MarkovTreeNode> MarkovTreeNodePtr;

  static std::shared_ptr<MarkovTreeNode> makeMarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent) {
    return std::shared_ptr<MarkovTreeNode>(new MarkovTreeNode(parent));
  }

  ptr<MarkovTreeNode> makeMarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent, const GraphicalModel& gm);

  /* add a child to the node */
  static MarkovTreeNodePtr addChild(MarkovTreeNodePtr node, const GraphicalModel& gm) {
    node->children.push_back(makeMarkovTreeNode(node, gm));
    return node->children.back();
  }

  struct MarkovTree {
  public:
    MarkovTree();
    std::shared_ptr<MarkovTreeNode> root; 

    // return log(sum(posterior weights of all nodes)).
    double logSumWeights(MarkovTreeNodePtr node); 
    // return log(sum(prior weights of all nodes)).
    double logSumPriorWeights(MarkovTreeNodePtr node, size_t max_level = -1);
    // return log expected reward starting from a node (include).
    double aggregateReward(MarkovTreeNodePtr node, double normalize, size_t max_level = -1);
    // return final tags.
    std::vector<std::shared_ptr<GraphicalModel> > aggregateSample(MarkovTreeNodePtr node, size_t max_level = -1);
    // return expected value of the gradient (unnormalized).
    std::pair<ParamPointer,double> aggregateGradient(std::shared_ptr<MarkovTreeNode> node, double normalize);
    // return expected gradient.
    ParamPointer expectedGradient();
    // return all samples.
    ptrs<GraphicalModel> getSamples();
    ptrs<GraphicalModel> getSamples(std::shared_ptr<MarkovTreeNode> node);

  };
}

#endif