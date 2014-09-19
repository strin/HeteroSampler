#pragma once

#include "utils.h"

namespace Tagging {

struct GraphicalModel {
public:
  GraphicalModel() {}
  virtual ~GraphicalModel() {}

  /* statistics for variables */
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

  /* randomness */
  objcokus* rng;  

  /* related reference and pointers */
  const Instance* seq;
  ptr<Corpus> corpus;

  // to string. 
  virtual string str() {
    throw "GraphicalModel::str not defined.";
  } 

  // get size of the graphical model.
  virtual size_t size() const {
    throw "GraphicalModel::size not defined.";
  }
};

}