#ifndef CORPUS_OPENGM
#define CORPUS_OPENGM

#include "utils.h"
#include "opengm.h"
#include "corpus.h"
#include "model.h"

#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

namespace Tagging {

template<class GM>
struct InstanceOpenGM : public Instance {
public:
  typedef GM GraphicalModelType;
  InstanceOpenGM(const Corpus* corpus, const GraphicalModelType& gm);
  GraphicalModelType gm;

  virtual void 
  parselines(const vec<string>& lines) {}

  virtual string 
  str() const {
    return "";
  };
};

template<class GM>
InstanceOpenGM<GM>::InstanceOpenGM(const Corpus* corpus, const GraphicalModelType& gm) 
: Instance(corpus), gm(gm) {

}

template<class GM>
struct CorpusOpenGM : public Corpus {
public:
  typedef GM GraphicalModelType;

  CorpusOpenGM() {}

  void read(const std::string& dirname, bool lets_shuffle = false);
};

template<class GM>
void CorpusOpenGM<GM>::read(const std::string& dirname, bool lets_shuffle) {
  DIR *dir;struct dirent *ent;
  if ((dir = opendir(dirname.c_str())) != NULL) {
    while((ent = readdir (dir)) != NULL) {
      GraphicalModelType instance;
      opengm::hdf5::load(instance, ent->d_name,"gm");
      this->seqs.push_back(ptr<InstanceOpenGM<GM> >(new InstanceOpenGM<GM>(this, instance)));
    }
    closedir (dir);
  }else{
    throw "CorpusOpenGM: could not open directory";
  }
}

}
#endif