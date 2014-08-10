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

unordered_map<string, StringVector> Model::word_feat;

Model::Model(const Corpus* corpus, int T, int B, int Q, double eta)
:corpus(corpus), param(makeParamPointer()),
  G2(makeParamPointer()) , stepsize(makeParamPointer()), 
  T(T), B(B), K(5), Q(Q), Q0(1),  
  testFrequency(0.3), eta(eta) {
  rngs.resize(K);
}

void Model::configStepsize(ParamPointer gradient, double new_eta) {
  for(const pair<string, double>& p : *gradient) 
    (*stepsize)[p.first] = new_eta;
}

/* use standard NLP functions of word */
StringVector Model::NLPfunc(const string word) {
  StringVector nlp = makeStringVector();
  nlp->push_back(word);
  /* unordered_map<std::string, StringVector>::iterator it = word_feat.find(word);
  if(it != word_feat.end())
    return it->second; */
  size_t wordlen = word.length();
  for(size_t k = 1; k <= 3; k++) {
    if(wordlen > k) {
      nlp->push_back("p"+to_string(k)+"-"+word.substr(0, k));
      nlp->push_back("s"+to_string(k)+"-"+word.substr(wordlen-k, k));
    }
  }
  if(std::find_if(word.begin(), word.end(), 
	  [](char c) { return std::isdigit(c); }) != word.end()) {
      nlp->push_back("00-");  // number
  }
  // word signature.
  stringstream sig0;
  string sig1(word);
  string lowercase = word;
  transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
  nlp->push_back(lowercase);
  char prev = '0';
  bool capitalized = true;
  for(size_t i = 0; i < wordlen; i++) {
    if(word[i] <= 'Z' && word[i] >= 'A') {
      if(prev != 'A') 
	sig0 << "A";
      sig1[i] = 'A';
      prev = 'A';
    }else if(word[i] <= 'z' && word[i] >= 'a') {
      if(prev != 'a')
	sig0 << "a";
      sig1[i] = 'a';
      prev = 'a';
      capitalized = false;
    }else{
      sig1[i] = 'x';
      prev = 'x';
      capitalized = false;
    }
  }
  nlp->push_back("SG-"+sig0.str());
  nlp->push_back("sg-"+sig1);
  if(capitalized) 
    nlp->push_back("CAP-");
  // word_feat[word] = nlp;
  return nlp;

}

tuple<FeaturePointer, double> Model::tagEntropySimple() const {
  FeaturePointer feat = makeFeaturePointer();
  const size_t taglen = corpus->tags.size();
  double logweights[taglen];
  // compute raw entropy.
  for(const pair<string, int>& p : corpus->dic) {
    auto count = corpus->word_tag_count.find(p.first);
    if(count == corpus->word_tag_count.end()) {
      cerr << "tagEntropy: word not found." << endl;
      (*feat)[p.first] = log(taglen);
      continue;
    }
    for(size_t t = 0; t < taglen; t++) {
      if(count->second[t] == 0)
	logweights[t] = -DBL_MAX;
      else
	logweights[t] = log(count->second[t]);
    }
    logNormalize(logweights, taglen);
    double entropy = 0.0;
    for(size_t t = 0; t < taglen; t++) {
      entropy -= logweights[t] * exp(logweights[t]);
    }
    (*feat)[p.first] = entropy;
  }
  // compute feature mean.
  double mean_ent = 0.0, count = 0;
  for(const Sentence& seq : corpus->seqs) {
    for(size_t i = 0; i < seq.seq.size(); i++) {
      mean_ent += (*feat)[seq.seq[i].word]; 
      count++;
    }
  }
  mean_ent /= count;
  // substract the mean.
  for(const pair<string, int>& p : corpus->dic) {
    (*feat)[p.first] -= mean_ent; 
  }
  return make_tuple(feat, mean_ent);
}

tuple<FeaturePointer, double> Model::wordFrequencies() const {
  FeaturePointer feat = makeFeaturePointer();
  // compute raw feature.
  for(const pair<string, int>& p : corpus->dic_counts) {
    (*feat)[p.first] = log(corpus->total_words)-log(p.second);
  }
  // compute feature mean.
  double mean_fre = 0, count = 0;
  for(const Sentence& seq : corpus->seqs) {
    for(size_t i = 0; i < seq.seq.size(); i++) {
      mean_fre += (*feat)[seq.seq[i].word];
      count++;
    }
  }
  mean_fre /= count;
  // substract the mean.
  for(const pair<string, int>& p : corpus->dic) {
    (*feat)[p.first] -= mean_fre;
  }
  return make_tuple(feat, mean_fre);
}

pair<Vector2d, vector<double> > Model::tagBigram() const {
  size_t taglen = corpus->tags.size();
  Vector2d mat = makeVector2d(taglen, taglen, 1.0);
  vector<double> vec(taglen, 1.0);
  for(const Sentence& seq : corpus->seqs) {
    vec[seq.tag[0]]++;
    for(size_t t = 1; t < seq.size(); t++) {
      mat[seq.tag[t-1]][seq.tag[t]]++; 
    }
  }
  for(size_t i = 0; i < taglen; i++) {
    vec[i] = log(vec[i])-log(taglen+corpus->seqs.size());
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

void Model::run(const Corpus& testCorpus, bool lets_test) {
  Corpus retagged(testCorpus);
  retagged.retag(*this->corpus); // use training taggs. 
  int testLag = corpus->seqs.size()*testFrequency;
  num_ob = 0;
  xmllog.begin("Q"); xmllog << Q << endl; xmllog.end();
  xmllog.begin("T"); xmllog << T << endl; xmllog.end();
  xmllog.begin("B"); xmllog << B << endl; xmllog.end();
  xmllog.begin("eta"); xmllog << eta << endl; xmllog.end();
  xmllog.begin("num_train"); xmllog << corpus->size() << endl; xmllog.end();
  xmllog.begin("num_test"); xmllog << testCorpus.size() << endl; xmllog.end();
  xmllog.begin("test_lag"); xmllog << testLag << endl; xmllog.end();
  for(int q = 0; q < Q; q++) {
    xmllog.begin("pass "+to_string(q));
    for(const Sentence& seq : corpus->seqs) {
      xmllog.begin("example_"+to_string(num_ob));
      ParamPointer gradient = this->gradient(seq);
      this->adagrad(gradient);
      xmllog.end();
      num_ob++;
      if(lets_test) {
	if(num_ob % testLag == 0) {
	  test(retagged);
	}
      }
    }
    xmllog.end();
  }
}

double Model::test(const Corpus& testCorpus) {
  Corpus retagged(testCorpus);
  retagged.retag(*this->corpus);
  int pred_count = 0, truth_count = 0, hit_count = 0;
  xmllog.begin("test");
  int ex = 0;
  XMLlog lg("error.xml");
  for(const Sentence& seq : retagged.seqs) {
    shared_ptr<Tag> tag = this->sample(seq).back();
    Tag truth(seq, corpus, &rngs[0], param);
    xmllog.begin("example_"+to_string(ex));
      xmllog.begin("truth"); xmllog << truth.str() << endl; xmllog.end();
      xmllog.begin("tag"); xmllog << tag->str() << endl; xmllog.end();
      xmllog.begin("dist"); xmllog << tag->distance(truth) << endl; xmllog.end();
    xmllog.end();
    bool pred_begin = false, truth_begin = false, hit_begin = false;
    for(int i = 0; i < seq.size(); i++) {
      if(corpus->mode == Corpus::MODE_POS) {
	if(tag->tag[i] == seq.tag[i]) {
	  hit_count++;
	}
	pred_count++;
      }else if(corpus->mode == Corpus::MODE_NER) {
	auto check_chunk_begin = [&] (const Tag& tag, int pos) {
	  string tg = corpus->invtags.find(tag.tag[pos])->second, 
		 prev_tg = pos > 0 ? corpus->invtags.find(tag.tag[pos-1])->second : "O";
	  string type = tag.seq->seq[pos].pos, 
		 prev_type = pos > 0 ? tag.seq->seq[pos-1].pos : "";
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
	  string tg = corpus->invtags.find(tag.tag[pos])->second, 
		 next_tg = pos < tag.size()-1 ? corpus->invtags.find(tag.tag[pos+1])->second : "O";
	  string type = tag.seq->seq[pos].pos, 
		 next_type = pos < tag.size()-1 ? tag.seq->seq[pos+1].pos : "";
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
	truth_count += (int)check_chunk_begin(truth, i);
	pred_count += (int)check_chunk_begin(*tag, i);
	if(check_chunk_begin(truth, i) && check_chunk_begin(*tag, i)) 
	  hit_begin = true;
	if(tag->tag[i] != seq.tag[i]) {
	  hit_begin = false;
	  lg.begin("error");
	  lg << corpus->invtags.find(truth.tag[i])->second << endl;
	  lg << corpus->invtags.find(tag->tag[i])->second << endl;
	  lg.end();
	}
	if(check_chunk_end(truth, i) && check_chunk_end(*tag, i)) { 
	  hit_count += (int)hit_begin;
	}
      }
    }
    ex++;
  }
  lg.end();
  xmllog.end();

  xmllog.begin("score"); 
  double accuracy = (double)hit_count/pred_count;
  double recall = (double)hit_count/truth_count;
  xmllog << "test precision = " << accuracy * 100 << " %" << endl; 
  if(corpus->mode == Corpus::MODE_POS) {
    xmllog.end();
    return accuracy;
  }else if(corpus->mode == Corpus::MODE_NER) {  
    xmllog << "test recall = " << recall * 100 << " %" << endl;
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    xmllog << "test f1 = " << f1 * 100 << " %" << endl;
    xmllog.end();
    return f1;
  }
  return -1;
}

void Model::sample(Tag& tag, int time) {
  tag = *this->sample(*tag.seq).back();
}

void Model::sampleOne(Tag& tag, int choice) {
  throw "custom kernel choice not implemented."; 
}

double Model::score(const Tag& tag) {
  return 0;
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

