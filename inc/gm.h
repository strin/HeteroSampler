#pragma once

#include "utils.h"
#include "corpus.h"

namespace Tagging {

class Location {
  public:
    int index, pos;
    Location() {}
    Location(int index, int pos)
    : index(index), pos(pos) {
    }
  };

class Value {
public:
  Location loc;
  double resp;
  Value() {}
  Value(Location loc, double resp)
  : loc(loc), resp(resp) {
  }
};

struct compare_value {
  bool operator()(const Value& n1, const Value& n2) const {
    return n1.resp < n2.resp;
  }
};

typedef boost::heap::fibonacci_heap<Value, boost::heap::compare<compare_value>> Heap;

struct GraphicalModel {
public:
  GraphicalModel() {
    time = 0;    
  }
  virtual ~GraphicalModel() {}

  /* statistics for variables */
  void initStats() {
    int num_tags = this->numLabels(0);
    entropy.resize(this->size(), log(num_tags));
    prev_entropy.resize(this->size());
    feat.resize(this->size());
    blanket.resize(this->size());
    changed.resize(this->size());
    handle.resize(this->size());
    feat.resize(this->size(), nullptr);

    sc.resize(num_tags, 0);
    prev_sc.resize(this->size());
    for(auto& s : prev_sc) {
      s.resize(num_tags, -log(num_tags));
    }
    this_sc.resize(this->size());
    for(auto& s : prev_sc) {
      s.resize(num_tags, -log(num_tags));
    }
  }

  int time;                               // how many times have spent on sampling this graphical model. 
  int oldval;                             // oldval before the latest sampling.
  vec<double> timestamp;                  // whenever a position is changed, its timestamp is incremented.
  vec<double> checksum;                   // if checksum is changes, then the position might be updated.
  std::vector<double> entropy;            // current entropy when being sampled.
  std::vector<double> prev_entropy;       // previous entropy before being sampled.
  std::vector<double> sc;                 // temporary normalized score.
  vec<vec<double> > this_sc, prev_sc;     // normalized score.
  std::vector<double> entropy_unigram;    // unigram entropy of positions.
  vec<vec<double> > sc_unigram;
  std::vector<double> reward;
  std::vector<double> resp;
  std::vector<int> mask;
  vec<map<int, int> > blanket;             // Markov blanket.
  vec<map<int, bool> > changed;                       // whether a location has changed.
  vec<typename Heap::handle_type> handle;
  std::vector<FeaturePointer> feat;

  /* randomness */
  objcokus* rng;  

  /* related reference and pointers */
  const Instance* seq;
  ptr<Corpus> corpus;

  // to string. 
  virtual string str(bool verbose = false) {
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

  // set the label of node *id*.
  virtual void setLabel(int id, int val) = 0;

  // get the labels of some nodes in blanket.
  virtual map<int, int> getLabels(vec<int> blanket) const {
    map<int, int> mb;
    for(auto id : blanket) {
      mb[id] = getLabel(id);
    }
    return mb;
  }

};

}