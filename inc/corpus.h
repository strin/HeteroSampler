#ifndef POS_SENTENCE_H
#define POS_SENTENCE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

struct Token {
public:
  std::string word,  // word. use lower case, B-S means beginning of sentence. 
	      pos, pos2, 
	      ner;   // name entity.
  bool is_doc_start;
  Token();
  Token(const std::string& line);
  void parseline(const std::string& line);
};

struct Corpus;

struct Sentence {
public:
  Sentence(const Corpus* const corpus);
  Sentence(const Corpus* const corpus, const std::vector<std::string>& lines);
  std::vector<Token> seq;
  std::vector<int> tag;
  const Corpus* const corpus;
  void parselines(const std::vector<std::string>& lines);
  std::string str() const;
  size_t size() const {return this->seq.size(); }
};

struct Corpus {
public:
  enum Mode {MODE_POS, MODE_NER};
  Mode mode;
  Corpus(Mode mode = MODE_POS);
  Corpus(const std::string& filename, Mode mode = MODE_POS);
  std::vector<Sentence> seqs;
  std::map<std::string, int> tags, tagcounts;
  std::map<std::string, int> dic, dic_counts;
  size_t total_words, total_tags;
  std::map<int, std::string> invtags;
  void read(const std::string& filename);
  void retag(const Corpus& corpus); // retag using corpus' tag.
  int size() const {return seqs.size(); }
};

#endif
