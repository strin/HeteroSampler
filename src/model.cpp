#include "model.h"
#include "utils.h"
#include "log.h"
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
  log.begin("truth"); log << seq.str() << endl; log.end();
  log.begin("tag"); log << tag.str() << endl; log.end();
  return gradient;
}

void Model::run(const Corpus& testCorpus) {
  Corpus retagged(testCorpus);
  retagged.retag(this->corpus); // use training taggs. 
  int testLag = corpus.seqs.size()*testFrequency;
  int numObservation = 0;
  log.begin("param");
  log.begin("Q"); log << Q << endl; log.end();
  log.begin("T"); log << T << endl; log.end();
  log.begin("B"); log << B << endl; log.end();
  log.begin("eta"); log << eta << endl; log.end();
  log.begin("num_train"); log << corpus.size() << endl; log.end();
  log.begin("num_test"); log << testCorpus.size() << endl; log.end();
  log.begin("test_lag"); log << testLag << endl; log.end();
  log.end();
  for(int q = 0; q < Q; q++) {
    log.begin("pass "+to_string(q));
    for(const Sentence& seq : corpus.seqs) {
      log.begin("example_"+to_string(numObservation));
      ParamPointer gradient = gradientGibbs(seq);
      this->adagrad(gradient);
      log.end();
      numObservation++;
      if(numObservation % testLag == 0) {
	log.begin("test");
	double f1 = test(retagged);
	log.end();
      }
    }
  }
}

double Model::test(const Corpus& corpus) {
  map<int, int> tagcounts;
  map<int, int> taghits;
  int testcount = 0, alltaghits = 0;
  log.begin("examples");
  for(const Sentence& seq : corpus.seqs) {
    Tag tag(&seq, corpus, &rngs[0], param);
    for(int t = 0; t < T; t++) {
      for(int i = 0; i < seq.tag.size(); i++) {
	tag.proposeGibbs(i);
      }
    }
    log.begin("truth"); log << seq.str() << endl; log.end();
    log.begin("tag"); log << tag.str() << endl; log.end();
    for(int i = 0; i < seq.tag.size(); i++) {
      if(tag.tag[i] == seq.tag[i]) {
	if(taghits.find(tag.tag[i]) == taghits.end())
	  taghits[tag.tag[i]] = 0;
	taghits[tag.tag[i]]++;
	alltaghits++;
      }
      if(tagcounts.find(tag.tag[i]) == tagcounts.end())
	tagcounts[tag.tag[i]] = 0;
      tagcounts[tag.tag[i]]++;
      testcount++;
    }
  }
  log.end();

  log.begin("score"); 
  double f1 = 0.0;
  for(const pair<string, int>& p : corpus.tags) {
    double accuracy = 0;
    if(tagcounts[p.second] != 0)
      accuracy = taghits[p.second]/(double)tagcounts[p.second];
    double recall = 0;
    if((double)corpus.tagcounts.find(p.first)->second != 0)
      recall = taghits[p.second]/(double)corpus.tagcounts.find(p.first)->second;
    log << "<tag: " << p.first << "\taccuracy: " << accuracy << "\trecall: " << recall << "\tF1: " <<
    2*accuracy*recall/(accuracy+recall) << endl;
    // if(accuracy != 0 && recall != 0)
    //  f1 += 2*accuracy*recall/(accuracy+recall);
  }
  double accuracy = (double)alltaghits/testcount;
  log << "test accuracy = " << accuracy*100 << " %" << endl; 
  log.end();
  return f1/corpus.tags.size();
}

void Model::adagrad(ParamPointer gradient) {
  for(const pair<string, double>& p : *gradient) {
    mapUpdate(*G2, p.first, p.second * p.second);
    mapUpdate(*param, p.first, eta * p.second / sqrt(1e-4 + (*G2)[p.first]));
  }
}
