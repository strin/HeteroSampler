#ifndef POS_FEAT_EXTRACT_H
#define POS_FEAT_EXTRACT_H

#include "utils.h"
#include "tag.h"


StringVector NLPfunc(const std::string word);
void extractUnigramFeature(const Tag& tag, int pos, int breadth, int depth, FeaturePointer output);
void extractBigramFeature(const Tag& tag, int pos, FeaturePointer output);

#endif
