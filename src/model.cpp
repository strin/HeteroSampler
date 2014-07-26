#include "model.h"
#include "utils.h"
#include <iostream>
#include <string>
#include <map>
#include <cmath>

using namespace std;

Model::Model(const Corpus& corpus)
:corpus(corpus), param(new map<string, double>()),
  G2(new map<string, double>()) {
  rngs.resize(numThreads);
}

ParamPointer Model::gradientGibbs(const Sentence& seq) {
  Tag tag(&seq, corpus, &rngs[0], param);
  FeaturePointer feat = tag.extractFeatures(seq.tag);
  ParamPointer gradient(new map<string, double>());
  for(int t = 0; t < T; t++) {
    for(int i = 0; i < seq.tag.size(); i++) 
      tag.proposeGibbs(i);
    if(t < B) continue;
    mapUpdate<double, int>(*gradient, *tag.extractFeatures(tag.tag));
  }
  mapDivide<double>(*gradient, -(double)(T-B));
  mapUpdate<double, int>(*gradient, *feat);
  cout << "[Example] " << tag.str() << endl;
  return gradient;
}

void Model::run(const Corpus& testCorpus) {
  int testLag = corpus.seqs.size()*testFrequency;
  int numObservation = 0;
  cout << "[run] corpus size = " << corpus.seqs.size() 
      << "\t Q = " << Q << endl;
  for(int q = 0; q < Q; q++) {
    cout << "iteration " << q << endl;
    for(const Sentence& seq : corpus.seqs) {
      cout << "pass " << numObservation/(double)corpus.seqs.size() << endl;
      ParamPointer gradient = gradientGibbs(seq);
      this->adagrad(gradient);
      numObservation++;
      if(numObservation % testLag == 0) {
	double f1 = test(testCorpus);
	cout << "test F1 score = " << f1*100 << " %" << endl;
      }
    }
  }
}

double Model::test(const Corpus& corpus) {
  map<int, int> tagcounts;
  map<int, int> taghits;
  int testcount = 0;
  for(const Sentence& seq : corpus.seqs) {
    Tag tag(&seq, corpus, &rngs[0], param);
    for(int t = 0; t < T; t++) {
      for(int i = 0; i < seq.tag.size(); i++) {
	tag.proposeGibbs(i);
      }
    }
    for(int i = 0; i < seq.tag.size(); i++) {
      if(tag.tag[i] == seq.tag[i]) {
	if(taghits.find(tag.tag[i]) == taghits.end())
	  taghits[tag.tag[i]] = 0;
	taghits[tag.tag[i]]++;
      }
      if(tagcounts.find(tag.tag[i]) == tagcounts.end())
	tagcounts[tag.tag[i]] = 0;
      tagcounts[tag.tag[i]]++;
    }
    testcount++;
  }
  double f1 = 0.0;
  for(const pair<string, int>& p : corpus.tags) {
    double accuracy = 0;
    if(tagcounts[p.second] != 0)
      accuracy = taghits[p.second]/(double)tagcounts[p.second];
    double recall = 0;
    if((double)corpus.tagcounts.find(p.first)->second != 0)
      recall = taghits[p.second]/(double)corpus.tagcounts.find(p.first)->second;
    cout << "<tag: " << p.first << "\taccuracy: " << accuracy << "\trecall: " << recall << endl;
    if(accuracy != 0 && recall != 0)
      f1 += 2*accuracy*recall/(accuracy+recall);
  }
  return f1/corpus.tags.size();
}

void Model::adagrad(ParamPointer gradient) {
  for(const pair<string, double>& p : *gradient) {
    mapUpdate(*G2, p.first, p.second * p.second);
    mapUpdate(*param, p.first, eta * p.second / sqrt(1e-4 + (*G2)[p.first]));
  }
}
