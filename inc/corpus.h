#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

#ifndef POS_SENTENCE_H
#define POS_SENTENCE_H

namespace Tagging {
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

  ///////// Instance //////////////////////////////////////////
  struct Instance {
  public:
    Instance(const Corpus* const corpus);
    Instance(const Corpus* const corpus, const std::vector<std::string>& lines);
    virtual void parselines(const std::vector<std::string>& lines) = 0;
    virtual std::string str() const = 0;
    virtual size_t size() const {return this->seq.size(); }

    std::vector<TokenPtr> seq;
    std::vector<int> tag;
    const Corpus* corpus;
  };
  typedef ptr<Instance> SentencePtr;

  //////// Token for POS/NER tagging ////////////////////////////////////
  struct TokenLiteral : public Token {
  public:
    // old convention, kept for compatibility.
    std::string word,  // word. use lower case, B-S means beginning of Instance. 
		pos, pos2, 
		ner;   // name entity.
    // new, more general convention.
    std::vector<std::string> token; // input.
    vec<int> itoken;             // token + sig.
    size_t depth() const {return token.size(); }    // return input depth.
    bool is_doc_start;

    TokenLiteral();
    TokenLiteral(const std::string& line);

    virtual void parseline(const std::string& line);
    virtual string str() const {return this->word; }
  };


  /////// Token for OCR tagging ///////////////////////////////////////////
  template<size_t height, size_t width>
  struct TokenOCR : public Token {
  public:
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

  /////// Instance //////////////////////////////////////////////////////
  struct SentenceLiteral : public Instance {
  public:
    SentenceLiteral(const Corpus* corpus);
    SentenceLiteral(const Corpus* corpus, const std::vector<std::string>& lines);

    virtual void parselines(const std::vector<std::string>& lines);
    virtual std::string str() const;
  };

  template<size_t height, size_t width>
  struct SentenceOCR : public Instance {
  public:
    SentenceOCR(const Corpus* corpus) : Instance(corpus) {};
    SentenceOCR(const Corpus* corpus, const std::vector<std::string>& lines)
      : Instance(corpus, lines) {
      this->parselines(lines);
    }
    
    virtual void parselines(const std::vector<string>& lines);
    virtual string str() const;
    virtual vec<int> markovBlanket(int id) const {
      vec<int> ret;
      if(id >= 1) ret.push_back(id-1);
      if(id < this->size()-1) ret.push_back(id+1);
      return ret;
    }
  };

  typedef std::shared_ptr<std::vector<std::string> > StringVector;
  static StringVector makeStringVector() {
    return StringVector(new std::vector<std::string>());
  }

  struct Corpus {
  public:
    Corpus() {test_count = -1; }
    std::vector<SentencePtr> seqs;

    objcokus cokus;
    double aveT;		  // average length of seq.

    map<string, int> tags;
    vec<string> invtags;

    std::string invtag(int id) const {
      assert(id < invtags.size());
      return invtags[id]; 
    }

    virtual void read(const std::string& filename, bool lets_shuffle = true) = 0;
    virtual void retag(ptr<Corpus> corpus); 

    int size() const {return seqs.size(); }
    int count(int test_count = -1) const {
      if(test_count < 0 || test_count >= seqs.size()) test_count = seqs.size();
      int c = 0;
      for(int i = 0; i < test_count; i++) {
        c += seqs[i]->size();
      }
      return c;
    }

    int test_count;
  };

  struct CorpusLiteral : public Corpus {
  public:
    CorpusLiteral();
    void read(const std::string& filename, bool lets_shuffle = true);
    void retag(ptr<Corpus> corpus);
    
    void computeWordFeat();  // compute and cache word features.
    StringVector getWordFeat(std::string word) const;

    /* stats utils */
    std::tuple<ParamPointer, double> tagEntropySimple() const;
    std::tuple<ParamPointer, double> wordFrequencies() const;
    std::pair<Vector2d, std::vector<double> > tagBigram() const;

    map<string, int> tagcounts;

    map<string, int> dic, dic_counts;
    vec<string> invdic;
    map<string, std::vector<int> > word_tag_count;

    size_t total_sig;
    size_t total_words;
  private:
    std::unordered_map<std::string, StringVector> word_feat;
    bool is_word_feat_computed;
  };

  template<size_t height, size_t width>
  struct CorpusOCR : public Corpus {
  public:
    CorpusOCR();
  
    void read(const std::string& filename, bool lets_shuffle = true);

    size_t h() const {return height; }
    size_t w() const {return width; }
  };

}


#endif
