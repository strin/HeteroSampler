#include "model.h"
#include "utils.h"
#include "log.h"
#include "MarkovTree.h"
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <thread>

using namespace std;

Model::Model(const Corpus& corpus)
:corpus(corpus), param(new map<string, double>()),
  G2(new map<string, double>()) ,
  T(1), B(0), K(5), Q(10), 
  testFrequency(0.3), eta(0.5) {
  rngs.resize(K);
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
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  return gradient;
}

ParamPointer Model::gradient(const Sentence& seq) {
  return gradientGibbs(seq);  
}

void Model::run(const Corpus& testCorpus) {
  Corpus retagged(testCorpus);
  retagged.retag(this->corpus); // use training taggs. 
  int testLag = corpus.seqs.size()*testFrequency;
  int numObservation = 0;
  xmllog.begin("param");
  xmllog.begin("Q"); xmllog << Q << endl; xmllog.end();
  xmllog.begin("T"); xmllog << T << endl; xmllog.end();
  xmllog.begin("B"); xmllog << B << endl; xmllog.end();
  xmllog.begin("eta"); xmllog << eta << endl; xmllog.end();
  xmllog.begin("num_train"); xmllog << corpus.size() << endl; xmllog.end();
  xmllog.begin("num_test"); xmllog << testCorpus.size() << endl; xmllog.end();
  xmllog.begin("test_lag"); xmllog << testLag << endl; xmllog.end();
  xmllog.end();
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
  }
}

double Model::test(const Corpus& corpus) {
  map<int, int> tagcounts;
  map<int, int> taghits;
  int testcount = 0, alltaghits = 0;
  xmllog.begin("examples");
  for(const Sentence& seq : corpus.seqs) {
    Tag tag(&seq, corpus, &rngs[0], param);
    for(int i = 0; i < seq.tag.size(); i++) {
      tag.proposeGibbs(i);
    }
    xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
    xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
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
  xmllog.end();

  xmllog.begin("score"); 
  double f1 = 0.0;
  for(const pair<string, int>& p : corpus.tags) {
    double accuracy = 0;
    if(tagcounts[p.second] != 0)
      accuracy = taghits[p.second]/(double)tagcounts[p.second];
    double recall = 0;
    if((double)corpus.tagcounts.find(p.first)->second != 0)
      recall = taghits[p.second]/(double)corpus.tagcounts.find(p.first)->second;
    xmllog << "<tag: " << p.first << "\taccuracy: " << accuracy << "\trecall: " << recall << "\tF1: " <<
    2*accuracy*recall/(accuracy+recall) << endl;
    // if(accuracy != 0 && recall != 0)
    //  f1 += 2*accuracy*recall/(accuracy+recall);
  }
  double accuracy = (double)alltaghits/testcount;
  xmllog << "test accuracy = " << accuracy*100 << " %" << endl; 
  xmllog.end();
  return f1/corpus.tags.size();
}

void Model::adagrad(ParamPointer gradient) {
  for(const pair<string, double>& p : *gradient) {
    mapUpdate(*G2, p.first, p.second * p.second);
    mapUpdate(*param, p.first, eta * p.second / sqrt(1e-4 + (*G2)[p.first]));
  }
}

ModelTreeUA::ModelTreeUA(const Corpus& corpus) 
:Model(corpus), eps(0), eps_split(0) {
}

ParamPointer ModelTreeUA::gradient(const Sentence& seq) {
  this->eps = 1.0/(T-B);
  MarkovTree tree;
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  std::function<void(int, shared_ptr<MarkovTreeNode>, Tag)> 
    core = [&](int id, shared_ptr<MarkovTreeNode> node, Tag tag) {
      objcokus cokus; // cokus is not re-entrant.
      cokus.seedMT(id);
      while(true) {
	node->gradient = tag.proposeGibbs(cokus.randomMT() % tag.size(), true);
	// xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
	if(node->depth < B) node->log_weight = -DBL_MAX;
	else node->log_weight = this->score(tag); 

	if(node == tree.root || cokus.random01() < this->eps_split) { // split.
	  vector<shared_ptr<thread> > th(K);
	  for(int k = 0; k < K; k++) {
	    node->children.push_back(makeMarkovTreeNode(node));
	    th[k] = shared_ptr<thread>(new thread(core, getFingerPrint((k+5)*3, id), 
			node->children.back(), tag)); 
	  }
	  for(int k = 0; k < K; k++) th[k]->join();
	  break;
	}else if(log(cokus.random01()) < log(this->eps)) { // stop.
	  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
	  break;
	}else{                             // proceed as chain.
	  node->children.push_back(makeMarkovTreeNode(node));
	  node = node->children.back();
	}
      }
  };
  Tag tag(&seq, corpus, &rngs[0], param);
  core(0, tree.root, tag);
  return tree.expectedGradient();
}

double ModelTreeUA::score(const Tag& tag) {
  const Sentence* seq = tag.seq;
  double score = 0.0;
  for(int i = 0; i < tag.size(); i++) {
    score -= (tag.tag[i] != seq->tag[i]);
  }
  return score;
}
