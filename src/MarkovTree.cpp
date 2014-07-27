#include "MarkovTree.h"

using namespace std;

MarkovTreeNode::MarkovTreeNode(shared_ptr<MarkovTreeNode> parent)
:parent(parent), log_weight(-DBL_MAX) {
  if(parent == nullptr) depth = 0;
  else depth = parent->depth+1;
}

MarkovTree::MarkovTree() 
:root(new MarkovTreeNode(nullptr)) {
}

double MarkovTree::logSumWeights(shared_ptr<MarkovTreeNode> node) {
  double weight = node->log_weight;
  for(shared_ptr<MarkovTreeNode> child : node->children) 
    weight = logAdd(weight, logSumWeights(child));
  return weight;
}

pair<ParamPointer, double> MarkovTree::aggregateGradient(shared_ptr<MarkovTreeNode> node, double normalize) {
  ParamPointer gradient = makeParamPointer();
  double weight = node->log_weight-normalize;
  for(shared_ptr<MarkovTreeNode> child : node->children) {
    pair<ParamPointer, double> result = aggregateGradient(child, normalize); 
    weight = logAdd(weight, result.second);
    mapUpdate(*gradient, *result.first);
  }
  mapUpdate(*gradient, *node->gradient, exp(weight));
  return pair<ParamPointer, double>(gradient, weight);
}

ParamPointer MarkovTree::expectedGradient() {
  return aggregateGradient(root, logSumWeights(root)).first;
}
