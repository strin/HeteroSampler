#include "model.h"
#include "utils.h"
#include "log.h"
#include "MarkovTree.h"
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <thread>
#include <chrono>

using namespace std;

Model::Model(const Corpus& corpus, int T, int B, int Q, double eta)
:corpus(corpus), param(new map<string, double>()),
  G2(new map<string, double>()) , stepsize(makeParamPointer()), 
  T(T), B(B), K(5), Q(Q), Q0(1),  
  testFrequency(0.3), eta(eta) {
  rngs.resize(K);
}

void Model::configStepsize(ParamPointer gradient, double new_eta) {
  for(const pair<string, double>& p : *gradient) 
    (*stepsize)[p.first] = new_eta;
}

FeaturePointer Model::tagEntropySimple() const {
  FeaturePointer feat = makeFeaturePointer();
  const size_t taglen = corpus.tags.size();
  double logweights[taglen];
  for(const pair<string, int>& p : corpus.dic) {
    for(size_t t = 0; t < taglen; t++) {
      string feat = "simple-"+p.first+"-"+to_string(t);
      logweights[t] = (*param)[feat];
    }
    logNormalize(logweights, taglen);
    double entropy = 0.0;
    for(size_t t = 0; t < taglen; t++) {
      entropy -= logweights[t] * exp(logweights[t]);
    }
    (*feat)[p.first] = entropy;
  }
  return feat;
}

FeaturePointer Model::wordFrequencies() const {
  FeaturePointer feat = makeFeaturePointer();
  for(const pair<string, int>& p : corpus.dic_counts) {
    (*feat)[p.first] = log(corpus.total_words)-log(p.second);
  }
  return feat;
}

pair<Vector2d, vector<double> > Model::tagBigram() const {
  size_t taglen = corpus.tags.size();
  Vector2d mat = makeVector2d(taglen, taglen, 1.0);
  vector<double> vec(taglen, 1.0);
  for(const Sentence& seq : corpus.seqs) {
    vec[seq.tag[0]]++;
    for(size_t t = 1; t < seq.size(); t++) {
      mat[seq.tag[t-1]][seq.tag[t]]++; 
    }
  }
  for(size_t i = 0; i < taglen; i++) {
    vec[i] = log(vec[i])-log(taglen+corpus.seqs.size());
    double sum_i = 0.0;
    for(size_t j = 0; j < taglen; j++) {
      sum_i += mat[i][j];
    }
    for(size_t j = 0; j < taglen; j++) {
      mat[i][j] = log(mat[i][j])-log(sum_i);
    }
  }
  return make_pair(mat, vec);
}

void Model::run(const Corpus& testCorpus) {
  Corpus retagged(testCorpus);
  retagged.retag(this->corpus); // use training taggs. 
  int testLag = corpus.seqs.size()*testFrequency;
  int numObservation = 0;
  xmllog.begin("Q"); xmllog << Q << endl; xmllog.end();
  xmllog.begin("T"); xmllog << T << endl; xmllog.end();
  xmllog.begin("B"); xmllog << B << endl; xmllog.end();
  xmllog.begin("eta"); xmllog << eta << endl; xmllog.end();
  xmllog.begin("num_train"); xmllog << corpus.size() << endl; xmllog.end();
  xmllog.begin("num_test"); xmllog << testCorpus.size() << endl; xmllog.end();
  xmllog.begin("test_lag"); xmllog << testLag << endl; xmllog.end();
  for(int q = 0; q < Q; q++) {
    xmllog.begin("pass "+to_string(q));
    for(const Sentence& seq : corpus.seqs) {
      xmllog.begin("example_"+to_string(numObservation));
      ParamPointer gradient = this->gradient(seq);
      this->adagrad(gradient);
      xmllog.end();
      numObservation++;
      if(numObservation % testLag == 0) {
	xmllog.begin("test");
	test(retagged);
	xmllog.end();
      }
    }
    xmllog.end();
  }
}

double Model::test(const Corpus& testCorpus) {
  int pred_count = 0, truth_count = 0, hit_count = 0;
  xmllog.begin("examples");
  for(const Sentence& seq : corpus.seqs) {
    shared_ptr<Tag> tag = this->sample(seq).back();
    xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
    xmllog.begin("tag"); xmllog << tag->str() << endl; xmllog.end();
    bool pred_begin = false, truth_begin = false, hit_begin = false;
    for(int i = 0; i < seq.size(); i++) {
      if(corpus.mode == Corpus::MODE_POS) {
	if(tag->tag[i] == seq.tag[i]) {
	  hit_count++;
	}
	pred_count++;
      }else if(corpus.mode == Corpus::MODE_NER) {
	string str_seq = corpus.invtags.find(seq.tag[i])->second, 
	       str_tag = corpus.invtags.find(tag->tag[i])->second;
	if(str_seq[0] == 'B') 
	  truth_begin = true;
	if(str_tag[0] == 'B')
	  pred_begin = true;
	if(str_seq[0] == 'B' && str_tag[0] == 'B')
	  hit_begin = true;
	if(tag->tag[i] != seq.tag[i]) hit_begin = false;
	string str_seq_next = corpus.invtags.find(seq.tag[i+1])->second, 
	       str_tag_next = corpus.invtags.find(tag->tag[i+1])->second;
	if(i == seq.size()-1 || str_seq_next == "O") {
	  truth_count += (truth_begin == true);
	  truth_begin = false;
	}
	if(i == seq.size()-1 || str_tag_next == "O") { 
	  pred_count += (pred_begin == true);
	  pred_begin = false;
	}
	if(i == seq.size()-1 || (str_seq_next == "O"
			     && str_tag_next == "O")) {
	  hit_count += (hit_begin == true);
	  hit_begin = false;
	}
      }
    }
  }
  xmllog.end();

  xmllog.begin("score"); 
  double accuracy = (double)hit_count/pred_count;
  double recall = (double)hit_count/truth_count;
  xmllog << "test precision = " << accuracy * 100 << " %" << endl; 
  if(corpus.mode == Corpus::MODE_POS) {
    xmllog.end();
    return accuracy;
  }else if(corpus.mode == Corpus::MODE_NER) {  
    xmllog << "test recall = " << recall * 100 << " %" << endl;
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    xmllog << "test f1 = " << f1 * 100 << " %" << endl;
    xmllog.end();
    return f1;
  }
  return -1;
}

void Model::adagrad(ParamPointer gradient) {
  for(const pair<string, double>& p : *gradient) {
    mapUpdate(*G2, p.first, p.second * p.second);
    double this_eta = eta;
    if(stepsize->find(p.first) != stepsize->end()) {
      this_eta = (*stepsize)[p.first];
    }
    mapUpdate(*param, p.first, this_eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
  }
}

