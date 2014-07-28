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

Model::Model(const Corpus& corpus)
:corpus(corpus), param(new map<string, double>()),
  G2(new map<string, double>()) ,
  T(1), B(0), K(5), Q(10), Q0(1),  
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

ParamPointer Model::gradientSimple(const Sentence& seq) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag oldtag(tag);
  ParamPointer gradient(new map<string, double>());
  for(size_t i = 0; i < tag.size(); i++) {
    ParamPointer g = tag.proposeSimple(i, true);
    mapUpdate<double, double>(*gradient, *g);
    mapUpdate<double, int>(*gradient, *tag.extractSimpleFeatures(tag.tag, i), -1.0);
    mapUpdate<double, int>(*gradient, *tag.extractSimpleFeatures(seq.tag, i));
  }
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  return gradient;
}

ParamPointer Model::gradient(const Sentence& seq) {
  return gradientGibbs(seq);  
}

void Model::runSimple(const Corpus& testCorpus) {
  Corpus retagged(testCorpus);
  retagged.retag(this->corpus); // use training taggs. 
  xmllog.begin("train_simple");
  int numObservation = 0;
  for(int q = 0; q < Q0; q++) {
    for(const Sentence& seq : corpus.seqs) {
      xmllog.begin("example_"+to_string(numObservation));
      ParamPointer gradient = this->gradientSimple(seq);
      this->adagrad(gradient);
      xmllog.end();
      numObservation++;
    }
  }
  copyParamFeatures(param, "simple-", "");
  xmllog.end();
  xmllog.begin("test");
  test(retagged);
  xmllog.end();
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
    xmllog.end();
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

ModelTreeUA::ModelTreeUA(const Corpus& corpus, int K) 
:Model(corpus), eps(0), eps_split(0) {
  Model::K = K;
  this->initThreads(K);
}

void ModelTreeUA::workerThreads(int id, shared_ptr<MarkovTreeNode> node, Tag tag, objcokus rng) {
    while(true) {
      node->gradient = tag.proposeGibbs(rng.randomMT() % tag.size(), true);
      // xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
      if(node->depth < B) node->log_weight = -DBL_MAX;
      else node->log_weight = this->score(tag); 

      if(node->depth == 0) { // multithread split.
	unique_lock<mutex> lock(th_mutex);
	active_work--;
	vector<objcokus> cokus(K);
	for(int k = 0; k < K; k++) {
	  int newid = getFingerPrint((k+5)*3, id);
	  cokus[k].seedMT(newid);
	  node->children.push_back(makeMarkovTreeNode(node));
	  th_work.push_back(make_tuple(newid, node->children.back(), tag, cokus[k]));
	}
	th_cv.notify_all();
	lock.unlock();
	return;
      }else if(log(rng.random01()) < log(this->eps_split)) {
	for(int k = 0; k < K; k++) {
	  node->children.push_back(makeMarkovTreeNode(node));
	  workerThreads(id, node->children.back(), tag, rng);
	}
	return;
      }else if(node->depth >= B && log(rng.random01()) < log(this->eps)) { // stop.
	unique_lock<mutex> lock(th_mutex);
	xmllog.begin("tag"); 
	xmllog << tag.str() << endl; 
	xmllog << "weight " << node->log_weight << endl;
	xmllog.end();
	active_work--;
	lock.unlock();
	return;
      }else{                             // proceed as chain.
	node->children.push_back(makeMarkovTreeNode(node));
	node = node->children.back();
      }
    }
}

void ModelTreeUA::initThreads(size_t numThreads) {
  th.resize(numThreads);
  active_work = 0;
  for(size_t ni = 0; ni < numThreads; ni++) {
    this->th[ni] = shared_ptr<thread>(new thread([&] (int id) {
      unique_lock<mutex> lock(th_mutex);
      while(true) {
	if(th_work.size() > 0) {
	  tuple<int, shared_ptr<MarkovTreeNode>, Tag, objcokus> work = th_work.front();
	  th_work.pop_front();
	  active_work++;
	  lock.unlock();
	  workerThreads(get<0>(work), get<1>(work), get<2>(work), get<3>(work));
	  th_finished.notify_all();
	  lock.lock();
	}else{
	  th_cv.wait(lock);	
	}
      }
    }, ni));
  }
}

ParamPointer ModelTreeUA::gradient(const Sentence& seq) {
  this->eps = 1.0/(T-B);
  MarkovTree tree;
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  Tag tag(&seq, corpus, &rngs[0], param);
  objcokus cokus; // cokus is not re-entrant.
  cokus.seedMT(0);
  unique_lock<mutex> lock(th_mutex);
  th_work.push_back(make_tuple(0, tree.root, tag, cokus));
  th_cv.notify_one();
  while(true) {
    th_finished.wait(lock);
    if(active_work + th_work.size() == 0) break;
  }
  lock.unlock();
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

ModelIncrGibbs::ModelIncrGibbs(const Corpus& corpus) 
:Model(corpus){
  
}

ParamPointer ModelIncrGibbs::gradient(const Sentence& seq) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag mytag(tag);
  FeaturePointer feat = tag.extractFeatures(seq.tag);
  ParamPointer gradient(new map<string, double>());
  for(int i = 0; i < seq.tag.size(); i++) {
    ParamPointer g = tag.proposeGibbs(i, true);
    mapUpdate<double, double>(*gradient, *g);
    mapUpdate<double, int>(*gradient, *tag.extractFeatures(tag.tag), -1);
    mytag.tag[i] = tag.tag[i];
    tag.tag[i] = seq.tag[i];
    mapUpdate<double, int>(*gradient, *tag.extractFeatures(tag.tag)); 
  }
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << mytag.str() << endl; xmllog.end();
  return gradient;
}

