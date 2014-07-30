#include "MarkovTree.h"

using namespace std;

MarkovTreeNode::MarkovTreeNode(shared_ptr<MarkovTreeNode> parent)
:parent(parent), log_weight(-DBL_MAX) {
  if(parent == nullptr) depth = 0;
  else depth = parent->depth+1;
  gradient = posgrad = neggrad = nullptr;
  tag = nullptr;
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
  double weight = node->log_weight-normalize, weight_descendant = -DBL_MAX;
  for(shared_ptr<MarkovTreeNode> child : node->children) {
    pair<ParamPointer, double> result = aggregateGradient(child, normalize); 
    weight_descendant = logAdd(weight_descendant, result.second);
    mapUpdate(*gradient, *result.first);
  }
  if(node->posgrad != nullptr)
    mapUpdate(*gradient, *node->posgrad, exp(weight));
  if(node->neggrad != nullptr)
    mapUpdate(*gradient, *node->neggrad, exp(weight_descendant));
  mapUpdate(*gradient, *node->gradient, exp(logAdd(weight, weight_descendant)));
  return pair<ParamPointer, double>(gradient, weight);
}

ParamPointer MarkovTree::expectedGradient() {
  return aggregateGradient(root, logSumWeights(root)).first;
}

TagVector MarkovTree::getSamples(shared_ptr<MarkovTreeNode> node) { 
  if(node->children.size() > 0) {
    TagVector vec;
    for(auto child : node->children) {
      TagVector res = getSamples(child); 
      vec.insert(vec.begin(), res.begin(), res.end()); 
    }
    return vec;
  }
  TagVector vec;
  vec.push_back(node->tag);
  return vec;
}

TagVector MarkovTree::getSamples() {
  return this->getSamples(this->root);
}
