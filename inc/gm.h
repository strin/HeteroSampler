#pragma once

#include "utils.h"

namespace Tagging {

struct GraphicalModel {
public:
  /* statistics for variables */
  std::vector<int> tag;
  vec<double> timestamp;                  // whenever a position is changed, its timestamp is incremented.
  vec<double> checksum;                   // if checksum is changes, then the position might be updated.
  std::vector<double> entropy;            // current entropy when sampled.
  std::vector<double> sc;                 
  std::vector<double> entropy_unigram;    // unigram entropy of positions.
  vec<vec<double> > sc_unigram;
  std::vector<double> reward;
  std::vector<double> resp;
  std::vector<int> mask;
  std::vector<FeaturePointer> feat;
  
};

}