#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

#ifndef POS_SENTENCE_H
#define POS_SENTENCE_H

namespace Tagging {
  /////////////////////////////////////////////////////////////////////
  /////// Token Interface /////////////////////////////////////////////
  struct Token {
  public:
    Token();
    Token(const std::string& line);
    virtual void parseline(const std::string& line) = 0;
    virtual string str() const = 0;

    std::string tag;                // output.
  };
  typedef ptr<Token> TokenPtr;


  struct Corpus;

  struct Sentence {
  public:
    Sentence(const Corpus* const corpus);
    Sentence(const Corpus* const corpus, const std::vector<std::string>& lines);
    virtual void parselines(const std::vector<std::string>& lines) = 0;
    virtual std::string str() const = 0;
    virtual size_t size() const {return this->seq.size(); }

    std::vector<TokenPtr> seq;
    std::vector<int> tag;
    const Corpus* corpus;
  };
  typedef ptr<Sentence> SentencePtr;

  ////////////////////////////////////////////////////////////////////////
  //////// Token for POS/NER tagging ////////////////////////////////////
  struct TokenLiteral : public Token {
  public:
    // old convention, kept for compatibility.
    std::string word,  // word. use lower case, B-S means beginning of sentence. 
		pos, pos2, 
		ner;   // name entity.
    // new, more general convention.
    std::vector<std::string> token; // input.
    size_t depth() const {return token.size(); }    // return input depth.
    bool is_doc_start;

    TokenLiteral();
    TokenLiteral(const std::string& line);

    virtual void parseline(const std::string& line);
    virtual string str() const {return this->word; }
  };


  //////////////////////////////////////////////////////////////////////////
  /////// Token for OCR tagging ///////////////////////////////////////////
  template<size_t height, size_t width>
  struct TokenOCR : public Token {
  public:
    std::string tag;
    TokenOCR();
    TokenOCR(const std::string& line);
    virtual void parseline(const std::string& line);
    // get pixel at row i, column j.
    const char get(int i, int j) {
      if(i >= height || j >= width || i < 0 || j < 0)
	throw "ocr access out of bound"; 
      return this->img[i][j]; 
    }
    virtual string str() const;
    int fold; // fold for cross validation.
  private:
    int img[height][width];
  };

  ///////////////////////////////////////////////////////////////////////
  /////// Sentence //////////////////////////////////////////////////////

  struct SentenceLiteral : public Sentence {
  public:
    SentenceLiteral(const Corpus* corpus);
    SentenceLiteral(const Corpus* corpus, const std::vector<std::string>& lines);

    virtual void parselines(const std::vector<std::string>& lines);
    virtual std::string str() const;
  };

  template<size_t height, size_t width>
  struct SentenceOCR : public Sentence {
  public:
    SentenceOCR(const Corpus* corpus) : Sentence(corpus) {};
    SentenceOCR(const Corpus* corpus, const std::vector<std::string>& lines)
      : Sentence(corpus, lines) {}
    
    virtual void parselines(const std::vector<string>& lines);
    virtual string str() const;
  };

  typedef std::shared_ptr<std::vector<std::string> > StringVector;
  static StringVector makeStringVector() {
    return StringVector(new std::vector<std::string>());
  }

  struct Corpus {
  public:
    Corpus();
    std::vector<SentencePtr> seqs;
    objcokus cokus;
    std::map<std::string, int> tags, tagcounts;
    std::map<std::string, int> dic, dic_counts;
    std::map<std::string, std::vector<int> > word_tag_count;
    size_t total_words;
    double aveT; // average length of seq.
    // total_tags;
    std::map<int, std::string> invtags;
    std::string invtag(int id) const {return invtags.find(id)->second; }

    template<class ST>
    void read(const std::string& filename, bool lets_shuffle = true) {
      std::ifstream file;
      file.open(filename);
      if(!file.is_open()) 
	throw "failed to read corpus file. ";
      std::string line;
      std::vector<std::string> lines;
      int tagid = 0;
      total_words = 0;
      this->seqs.clear();
      this->tags.clear();
      while(!file.eof()) {
	getline(file, line);
	if(line == "") {
	  SentencePtr sen = SentencePtr(new ST(this, lines));
	  lines.clear();
	  if(sen->seq.size() > 0) {
	    seqs.push_back(sen);
	  }else continue;
	  for(const TokenPtr token : seqs.back()->seq) {
	    std::string tg;
	    tg = token->tag;
	    if(tags.find(tg) == tags.end()) {
	      tags[tg] =  tagid++;
	      tagcounts[tg] = 0;
	    }
	    tagcounts[tg]++;
	    if(std::dynamic_pointer_cast<TokenLiteral>(token) != NULL) {
	      ptr<TokenLiteral> token_literal = std::dynamic_pointer_cast<TokenLiteral>(token);
	      dic[token_literal->word] = true;
	      if(dic_counts.find(token_literal->word) == dic_counts.end()) 
		dic_counts[token_literal->word] = 0;
	      dic_counts[token_literal->word]++;
	    }
	    total_words++;
	  }
	  continue;
	}else
	  lines.push_back(line);
      }
      // convert raw tag into integer tag.
      invtags.clear();
      word_tag_count.clear();
      aveT = 0;
      for(SentencePtr seq : seqs) {
	aveT += seq->size();
	for(const TokenPtr token : seq->seq) {
	  std::string tg;
	  tg = token->tag;
	  int itg = tags[tg];
	  seq->tag.push_back(itg); 
	  invtags[itg] = tg;
	  if(isinstance<TokenLiteral>(token)) {
	    ptr<TokenLiteral> token_literal = std::dynamic_pointer_cast<TokenLiteral>(token);
	    if(word_tag_count.find(token_literal->word) == word_tag_count.end()) {
	      word_tag_count[token_literal->word].resize(tagid, 0.0);
	    }
	    word_tag_count[token_literal->word][itg]++;
	  }
	}
      }
      aveT /= (double)seqs.size();
      // shuffle corpus.
      if(lets_shuffle)
	shuffle<SentencePtr>(seqs, cokus);
    }

    void retag(const Corpus& corpus); // retag using corpus' tag.
    int size() const {return seqs.size(); }
    void computeWordFeat();  // compute and cache word features.
    StringVector getWordFeat(std::string word) const;

  private:
    std::unordered_map<std::string, StringVector> word_feat;
    bool is_word_feat_computed;
  };

  // explicitly init 16 x 8 template.
  template class TokenOCR<16, 8>;
  template class SentenceOCR<16, 8>;
}
#endif
