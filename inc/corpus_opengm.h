#ifndef CORPUS_OPENGM
#define CORPUS_OPENGM

#include "utils.h"
#include "opengm.h"
#include "corpus.h"
#include "model.h"

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

namespace HeteroSampler {

template<class GM>
struct InstanceOpenGM : public Instance {
public:
  typedef GM GraphicalModelType;
  InstanceOpenGM(const Corpus* corpus, ptr<GraphicalModelType> gm);
  ptr<GraphicalModelType> gm;

  virtual void 
  parselines(const vec<string>& lines) {}

  virtual string 
  str() const {
    return "";
  };

  virtual size_t size() const {
    return gm->numberOfVariables();
  }

  vec<int> markovBlanket(int id) const {
    auto adj_set = adjacencyList[id];
    return vec<int>(adj_set.begin(), adj_set.end());
  }

  vec<int> invMarkovBlanket(int id) const {
    return markovBlanket(id);
  }

private:
  vec<set<size_t> > adjacencyList;
};

template<class GM>
InstanceOpenGM<GM>::InstanceOpenGM(const Corpus* corpus, ptr<GraphicalModelType> gm) 
: Instance(corpus), gm(gm) {
  gm->variableAdjacencyList(this->adjacencyList);
}

template<class GM>
struct CorpusOpenGM : public Corpus {
public:
  typedef GM GraphicalModelType;

  CorpusOpenGM() {}

  void read(const std::string& dirname, bool lets_shuffle = false);

  virtual void retag(ptr<Corpus> corpus) {}
};

template<class GM>
void CorpusOpenGM<GM>::read(const std::string& dirname, bool lets_shuffle) {
  DIR *dir;struct dirent *ent;
  if ((dir = opendir(dirname.c_str())) != NULL) {
    while((ent = readdir (dir)) != NULL) {
      ptr<GraphicalModelType> instance = std::make_shared<GraphicalModelType>();
      if(ent->d_name[0] == '.') continue;
      opengm::hdf5::load(*instance, dirname+"/"+ent->d_name,"gm");
      this->seqs.push_back(ptr<InstanceOpenGM<GM> >(new InstanceOpenGM<GM>(this, instance)));
    }
    closedir (dir);
  }else{
    throw "CorpusOpenGM: could not open directory";
  }
}

}
#endif
