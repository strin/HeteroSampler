#ifndef POS_MODEL_H
#define POS_MODEL_H

#include "tag.h"
#include "gm.h"
#include "corpus.h"
#include "MarkovTree.h"

namespace Tagging {
  inline static void adagrad(ParamPointer param, ParamPointer G2, ParamPointer gradient, double eta) {
    for(const std::pair<std::string, double>& p : *gradient) {
      mapUpdate(*G2, p.first, p.second * p.second);
      mapUpdate(*param, p.first, eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
    }
  }

  struct Model {
  public:
    Model(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);
    virtual void run(ptr<Corpus> test_corpus, bool lets_test = true);
    double test(ptr<Corpus> test_corpus);

    /* gradient interface */
    virtual ParamPointer gradient(const Instance& seq) = 0; 
    virtual TagVector sample(const Instance& seq, bool argmax = false) = 0;
    // infer under computational constraints.
    // emulate t-step transition of a markov chain.
    // default: use Model::sample(*tag.seq), i.e. time = T.
    virtual void sample(Tag& tag, int time, bool argmax = false);             // inplace.

    // sample using custom kernel choice.
    // return: gradient.
    virtual ParamPointer sampleOne(GraphicalModel& gm, objcokus& rng, int choice);           

    // sample using custom kernel choice at initialization.
    // only applies if "init" flag is on (not equal to *random*). 
    virtual ParamPointer sampleOneAtInit(GraphicalModel& gm, objcokus& rng, int choice);           

    // <deprecated> score a tag ? 
    virtual double score(const Tag& tag);

    // evaluate the accuracy for POS tag aginst truth.
    // return 0: hit count.
    // return 1: pred count.
    std::tuple<int, int> evalPOS(const Tag& tag);
    // evaulate the F1 score for NER tag aginst truth.
    // return 0: hit count.
    // return 1: pred count.
    // return 2: truth count.
    std::tuple<int, int, int> evalNER(const Tag& tag);

    /* parameters */
    size_t T, B, Q;
    int Q0;
    double testFrequency;
    double eta;
    std::vector<objcokus> rngs;
    ptr<Corpus> corpus;
    ParamPointer param, G2, stepsize;   // model.

    /* IO */
    friend std::ostream& operator<<(std::ostream& os, const Model& model);
    friend std::istream& operator>>(std::istream& os, Model& model);

    XMLlog xmllog;
    virtual void logArgs(); 
    /* const environment */
    enum Scoring {SCORING_NER, SCORING_ACCURACY };
    Scoring scoring;
    const boost::program_options::variables_map& vm;

  protected:
    void adagrad(ParamPointer gradient);
    void configStepsize(FeaturePointer gradient, double new_eta);

    int K;          // num of particle. 
    int num_ob;     // current number of observations.
  };

  typedef std::shared_ptr<Model> ModelPtr;
  typedef std::function<FeaturePointer(ptr<Model> model, const GraphicalModel& tag, int pos)> FeatureExtractOne;
  typedef std::function<FeaturePointer(ptr<Model> model, const GraphicalModel& tag)> FeatureExtractAll;

  struct ModelSimple : public Model {
  public:
    ModelSimple(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);
    ParamPointer gradient(const Instance& seq, TagVector* vec = nullptr, bool update_grad = true);
    ParamPointer gradient(const Instance& seq);
    virtual TagVector sample(const Instance& seq, bool argmax = false); 
    virtual void sample(Tag& tag, int time, bool argmax = false);
    FeaturePointer extractFeatures(const Tag& tag, int pos);
    
    virtual void logArgs();

    int windowL, depthL; // range of unigram features.
  };

  struct ModelCRFGibbs : public ModelSimple, public std::enable_shared_from_this<Model> {
  public:
    ModelCRFGibbs(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);
    ParamPointer gradient(const Instance& seq, TagVector* vec = nullptr, bool update_grad = true);
    ParamPointer gradient(const Instance& seq);
    virtual TagVector sample(const Instance& seq, bool argmax = false);
    virtual void sample(Tag& tag, int time, bool argmax = false);
    
    double score(const Tag& tag);
    virtual void logArgs();

    /* interface for feature extraction. */    
    FeatureExtractOne extractFeatures;
    FeatureExtractOne extractFeaturesAtInit;
    FeatureExtractAll extractFeatAll;

    ParamPointer sampleOne(GraphicalModel& tag, objcokus& rng, int choice);
    ParamPointer sampleOneAtInit(GraphicalModel& tag, objcokus& rng, int choice);

    ParamPointer proposeGibbs(Tag& tag, objcokus& rng, int pos, FeatureExtractOne feat_extract, bool grad_expect, bool grad_sample);

    int factorL;
    
  private:
    ParamPointer sampleOne(GraphicalModel& gm, objcokus& rng, int choice, FeatureExtractOne feat_extract);

    FeaturePointer extractFeaturesAll(const Tag& tag);

    void sampleOneSweep(Tag& tag, bool argmax = false);
  };

  struct ModelTreeUA : public ModelCRFGibbs {
  public:
    ModelTreeUA(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);

    virtual void run(ptr<Corpus> test_corpus);

    virtual std::shared_ptr<MarkovTree> explore(const Instance& seq);
    virtual ParamPointer gradient(const Instance& seq);
    virtual TagVector sample(const Instance& seq);
    virtual double score(const Tag& tag);

    /* parameters */
    double eps, eps_split;

    /* parallel environment */
    virtual void workerThreads(int tid, std::shared_ptr<MarkovTreeNode> node, Tag tag);
    std::vector<std::shared_ptr<std::thread> > th;
    std::list<std::tuple<std::shared_ptr<MarkovTreeNode>, Tag> > th_work;
    size_t active_work;
    std::mutex th_mutex;
    std::condition_variable th_cv, th_finished;
    std::vector<std::shared_ptr<std::stringstream> > th_stream;
    std::vector<std::shared_ptr<XMLlog> > th_log;

  protected:
    int Q0;
    void initThreads(size_t numThreads);
  };


  struct ModelAdaTree : public ModelTreeUA {
  public:
    ModelAdaTree(ptr<Corpus> corpus, const boost::program_options::variables_map& vm);
    /* implement components necessary */  
    void workerThreads(int tid, MarkovTreeNodePtr node, Tag tag);
    /* extract posgrad and neggrad for stop-or-not logistic regression */
    std::tuple<double, ParamPointer, ParamPointer, FeaturePointer> logisticStop
      (MarkovTreeNodePtr node, const Instance& seq, const Tag& tag); 

    /* stop feature extraction for each word */
    virtual FeaturePointer extractStopFeatures
      (MarkovTreeNodePtr node, const Instance& seq, const Tag& tag, int pos);
    /* stop feature extraction for entire Instance */
    virtual FeaturePointer extractStopFeatures
      (MarkovTreeNodePtr node, const Instance& seq, const Tag& tag);

    double score(MarkovTreeNodePtr node, const Tag& tag);

    /* parameters */
    double etaT;
  private:
    ParamPointer wordent, wordfreq;
    double wordent_mean, wordfreq_mean;
    Vector2d tag_bigram;
    std::vector<double> tag_unigram_start;
    double m_c, m_Tstar;
  };
}
#endif
