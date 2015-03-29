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
namespace po = boost::program_options;

namespace Tagging {
  Model::Model(ptr<Corpus> corpus, const po::variables_map& vm)
  :corpus(corpus), param(makeParamPointer()), vm(vm),
   G2(makeParamPointer()) , stepsize(makeParamPointer()), 
   T(vm["T"].as<size_t>()), B(vm["B"].as<size_t>()), 
   Q(vm["Q"].as<size_t>()), eta(vm["eta"].as<double>()), 
   K(vm["K"].as<size_t>()), 
   testFrequency(vm["testFrequency"].as<double>()) {
    try {
      if(vm.count("scoring") == 0) {
        throw "no scoring specified";
      }else{
        string scoring_str = vm["scoring"].as<string>();
        if(scoring_str == "Acc") scoring = SCORING_ACCURACY;
        else if(scoring_str == "NER") scoring = SCORING_NER;
        else if(scoring_str == "Lhood") scoring = SCORING_LHOOD;
        else throw "scoring method invalid";
      }
    }catch(char const* warn) {
      cout << warn << " - use accuracy" << endl;
      scoring = SCORING_ACCURACY;
    }
    rngs.resize(K);
  }

  void Model::configStepsize(FeaturePointer gradient, double new_eta) {
    for(const pair<string, double>& p : *gradient) 
      (*stepsize)[p.first] = new_eta;
  }


  void Model::run(ptr<Corpus> test_corpus, bool lets_test) {
    test_corpus->retag(this->corpus);
    int testLag = corpus->seqs.size()*testFrequency;
    num_ob = 0;
    this->logArgs();
    xmllog.begin("num_train"); xmllog << corpus->size() << endl; xmllog.end();
    xmllog.begin("num_test"); xmllog << test_corpus->size() << endl; xmllog.end();
    xmllog.begin("test_lag"); xmllog << testLag << endl; xmllog.end();
    for(int q = 0; q < Q; q++) {
      xmllog.begin("pass "+to_string(q));
      for(const SentencePtr seq : corpus->seqs) {
        xmllog.begin("example_"+to_string(num_ob));
        ParamPointer gradient = this->gradient(*seq);
        this->adagrad(gradient);
        xmllog.end();
        num_ob++;
        if(lets_test) {
          if(num_ob % testLag == 0) {
            test(test_corpus);
          }
        }
      }
      xmllog.end();
    }
  }

  void Model::logArgs() {
    xmllog.begin("Q"); xmllog << Q << endl; xmllog.end();
    xmllog.begin("T"); xmllog << T << endl; xmllog.end();
    xmllog.begin("B"); xmllog << B << endl; xmllog.end();
    xmllog.begin("eta"); xmllog << eta << endl; xmllog.end();
    xmllog.begin("taglen"); xmllog << corpus->tags.size() << endl; xmllog.end();
  }

  tuple<int, int> Model::evalPOS(const Tag& tag) {
    Tag truth(*tag.seq, corpus, &rngs[0], param);
    int hit_count = 0, pred_count = 0;
    for(int i = 0; i < truth.size(); i++) {
      if(tag.tag[i] == truth.tag[i]) {
        hit_count++;
      }
      pred_count++;
    }
    return make_tuple(hit_count, pred_count);
  }

  tuple<int, int, int> Model::evalNER(const Tag& tag) {
    const SentenceLiteral* seq = (const SentenceLiteral*)tag.seq;
    Tag truth(*tag.seq, corpus, &rngs[0], param);
    int hit_count = 0, pred_count = 0, truth_count = 0;
    auto check_chunk_begin = [&] (const Tag& tag, int pos) {
      string tg = tag.getTag(pos), 
             prev_tg = pos > 0 ? tag.getTag(pos-1) : "O";
      string type = cast<TokenLiteral>(seq->seq[pos])->pos, 
             prev_type = pos > 0 ? cast<TokenLiteral>(seq->seq[pos-1])->pos : "";
      char tg_ch = tg[0], prev_tg_ch = prev_tg[0];  
      return (prev_tg_ch == 'B' && tg_ch == 'B') ||
             (prev_tg_ch == 'I' && tg_ch == 'B') ||
             (prev_tg_ch == 'O' && tg_ch == 'B') ||
             (prev_tg_ch == 'O' && tg_ch == 'I') ||
             (prev_tg_ch == 'E' && tg_ch == 'E') ||
             (prev_tg_ch == 'E' && tg_ch == 'I') ||
             (prev_tg_ch == 'O' && tg_ch == 'E') ||
             (prev_tg_ch == 'O' && tg_ch == 'I') ||
             (tg != "O" && tg != "." && type != prev_type) ||
             (tg == "[") || (tg == "]");
    };
    auto check_chunk_end = [&] (const Tag& tag, int pos) { 
      string tg = tag.getTag(pos), 
             next_tg = pos < tag.size()-1 ? tag.getTag(pos+1) : "O";
      string type = cast<TokenLiteral>(seq->seq[pos])->pos, 
             next_type = pos < tag.size()-1 ? cast<TokenLiteral>(seq->seq[pos+1])->pos : "";
      char tg_ch = tg[0], next_tg_ch = next_tg[0];
      return (tg_ch == 'B' && next_tg_ch == 'B') ||
             (tg_ch == 'B' && next_tg_ch == 'O') ||
             (tg_ch == 'I' && next_tg_ch == 'B') ||
             (tg_ch == 'I' && next_tg_ch == 'O') ||
             (tg_ch == 'E' && next_tg_ch == 'E') ||
             (tg_ch == 'E' && next_tg_ch == 'I') ||
             (tg_ch == 'E' && next_tg_ch == 'O') ||
             (tg != "O" && tg != "." && type != next_type) ||
             (tg == "[") || (tg == "]");

    };
    bool hit_begin = true;
    for(int i = 0; i < truth.size(); i++) {
      truth_count += (int)check_chunk_begin(truth, i);
      pred_count += (int)check_chunk_begin(tag, i);
      if(check_chunk_begin(truth, i) && check_chunk_begin(tag, i)) 
        hit_begin = true;
      if(tag.tag[i] != truth.tag[i]) {
        hit_begin = false;
      }
      if(check_chunk_end(truth, i) && check_chunk_end(tag, i)) { 
        hit_count += (int)hit_begin;
      }
    }
    return make_tuple(hit_count, pred_count, truth_count);
  }

  double Model::test(ptr<Corpus> test_corpus) {
    test_corpus->retag(this->corpus);
    int pred_count = 0, truth_count = 0, hit_count = 0;
    xmllog.begin("test");
    int ex = 0;
    for(const SentencePtr seq : test_corpus->seqs) {
      TagVector tags = this->sample(*seq, false);
      double max_lhood = -DBL_MAX;
      shared_ptr<Tag> tag = tags.back();
      for(shared_ptr<Tag> this_tag : tags) {
        double sc = this->score(*this_tag);
        if(sc > max_lhood) {
          max_lhood = sc;
          tag = this_tag;
        }
      }
      Tag truth(*seq, corpus, &rngs[0], param);
      xmllog.begin("example_"+to_string(ex));
      // xmllog.begin("truth"); xmllog << truth.str() << endl; xmllog.end();
      xmllog.begin("truth"); xmllog << tag->seq->str() << endl; xmllog.end();
      xmllog.begin("tag"); xmllog << tag->str() << endl; xmllog.end();
      xmllog.begin("dist"); xmllog << tag->distance(truth) << endl; xmllog.end();
      xmllog.end();
      if(this->scoring == SCORING_ACCURACY) {
        tuple<int, int> hit_pred = this->evalPOS(*tag);
        hit_count += get<0>(hit_pred);
        pred_count += get<1>(hit_pred);
      }else if(this->scoring == SCORING_NER) {
        tuple<int, int, int> hit_pred_truth = this->evalNER(*tag);
        hit_count += get<0>(hit_pred_truth);
        pred_count += get<1>(hit_pred_truth);
        truth_count += get<2>(hit_pred_truth);
      }
      ex++;
    }
    xmllog.end();

    xmllog.begin("score"); 
    double accuracy = (double)hit_count/pred_count;
    double recall = (double)hit_count/truth_count;
    xmllog << "test precision = " << accuracy * 100 << " %" << endl; 
    if(this->scoring  == SCORING_ACCURACY) {
      xmllog.end();
      return accuracy;
    }else if(this->scoring == SCORING_NER) {  
      xmllog << "test recall = " << recall * 100 << " %" << endl;
      double f1 = 2 * accuracy * recall / (accuracy + recall);
      xmllog << "test f1 = " << f1 * 100 << " %" << endl;
      xmllog.end();
      return f1;
    }
    return -1;
  }

  void Model::sample(Tag& tag, int time, bool argmax) {
    tag = *this->sample(*tag.seq, argmax).back();
  }

  void Model::sampleOne(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature) {
    throw "custom kernel choice not implemented."; 
  }

  void Model::sampleOneAtInit(GraphicalModel& gm, objcokus& rng, int choice, bool use_meta_feature) {
    return this->sampleOne(gm, rng, choice);
  }

  double Model::score(const GraphicalModel& tag) {
    throw "Model::score not implemented.";
  }

  void Model::adagrad(ParamPointer gradient) {
    for(const pair<string, double>& p : *gradient) {
      mapUpdate(*G2, p.first, p.second * p.second);
      double this_eta = eta;
      if(stepsize->find(p.first) != stepsize->end()) {
        this_eta = (*stepsize)[p.first];
      }
      mapUpdate(*param, p.first, this_eta * p.second/sqrt(1 + (*G2)[p.first]));
    }
  }

  ostream& operator<<(ostream& os, const Model& model) {
    for(const pair<string, double>& p : *model.param) {
      os << p.first << " " << p.second << endl;
    }
    return os;
  }

  istream& operator>>(istream& is, Model& model) {
    string line;
    while(!is.eof()) {
      getline(is, line);
      if(line == "") break;
      vector<string> parts;
      split(parts, line, boost::is_any_of(" "));
      (*model.param)[parts[0]] = stod(parts[1]);
    }
    return is;
  }
}
