#pragma once

#include "utils.h"
#include "corpus.h"

namespace Tagging {

struct GraphicalModel {
public:
  GraphicalModel() {
    time = 0;    
  }
  virtual ~GraphicalModel() {}

  /* statistics for variables */
  int time;                               // how many times have spent on sampling this graphical model. 
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

  // get number of labels for variable i.
  virtual size_t numLabels(int id) const {
    throw "GraphicalModel::numLabels not defined.";
  }

  // get the label of node *id*.
  virtual int getLabel(int id) const = 0;

  // get the Markov blanket of a variable. 
  // default implementation returns all variables except id.
  virtual map<int, int> markovBlanket(int id) const {
    map<int, int> mb;
    for(int i = 0; i < this->size(); i++) {
      if(i == id) continue;
      mb[i] = getLabel(i);
    }
  }
};

}