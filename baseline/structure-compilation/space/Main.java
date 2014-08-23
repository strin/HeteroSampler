package space;

/**
 * This package is a collection of methods for training CRFs and maybe
 * max-margin methods in the future.
 *
 * Methods for optimization
 *  - Gradient with backtracking line search
 *  - L-BFGS with backtracking line search
 *  - Trust region Newton based on conjugate gradient (need Hessian)
 *  - Natural gradient ($O(d^2)$ per iteration)
 *  - Stochastic gradient
 *  - Stochastic meta-gradient
 *  - Averaged perceptron
 *  - Exponentiated gradient for max-margin/log-linear models
 *
 * Models
 *  - Binary logistic regression
 *  - Multiclass logistic regression
 *  - Conditional random fields
 *  - Binary SVMs
 *  - Max-margin models
 *
 * Representations
 *  - Primal
 *  - Dual
 * 
 * Assume sparse binary features.
 * Each position has a tuple of base predicate strings.
 * Each predicate is a tuple of (relation offset, base predicate).
 * Each feature is a conjunction of label and predicate.
 */

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class Main implements Runnable {
  @OptionSet(name="pred")
    public PredicateBank.Options predOptions = new PredicateBank.Options();
  @OptionSet(name="data") public Dataset.Options dataOptions = new Dataset.Options();
  @OptionSet(name="model") public Model.Options modelOptions = new Model.Options();
  @OptionSet(name="learn") public Learner.Options learnOptions = new Learner.Options();
  @OptionSet(name="blearn") public BatchLearner.Options blearnOptions = new BatchLearner.Options();
  @OptionSet(name="sglearn") public StochasticGradientLearner.Options sglearnOptions = new StochasticGradientLearner.Options();
  @OptionSet(name="eglearn") public ExponentiatedGradientLearner.Options eglearnOptions = new ExponentiatedGradientLearner.Options();

  public enum LearnerType { batch, sgd, exgd };
  @Option public LearnerType learnerType = LearnerType.batch;

  @Option public boolean computeEdgeMutualInfo = false;

  public void run() {
    // Load features and data
    begin_track("Setup");
    PredicateBank bank = new PredicateBank(predOptions);
    FeatureSet features = new FeatureSet(bank.getPredicates());
    Dataset dataset = new Dataset(dataOptions);
    dataset.load();
    Example.serializeStringData = false; // Don't to write strings any more
    features.extractPredicates(dataset.getTrainExamples());
    features.cachePredicates(dataset.getTrainExamples(), true);
    features.cachePredicates(dataset.getTestExamples(), true);
    end_track();

    // Learn the model
    Model model = new Model(modelOptions, features);
    Learner learner;
    switch(learnerType) {
      case batch: learner = new BatchLearner(learnOptions, blearnOptions, model); break;
      case sgd: learner = new StochasticGradientLearner(learnOptions, sglearnOptions, model); break;
      case exgd: learner = new ExponentiatedGradientLearner(learnOptions, eglearnOptions, model); break;
      default: throw Exceptions.unknownCase;
    }
    learner.setData(dataset.getTrainExamples(), dataset.getTestExamples());
    learner.learn();

    // Output train examples (true labels)
    ExampleWriter.write(Execution.getFile("train.examples"),
        model.features.tagIndexer, dataset.getTrainExamples());

    // Label test examples with system (predicted labels)
    learner.labelExamples(dataset.getTestExamples());
    ExampleWriter.write(Execution.getFile("test-pred.examples"),
        model.features.tagIndexer, dataset.getTestExamples());

    // Label unlabeled examples
    // (have to do it gradually, and no need to save cached state to disk!)
    ExampleWriter out = new ExampleWriter(
      Execution.getFile("unlabeled-pred.examples"), model.features.tagIndexer);
    begin_track("Labeling unlabeled examples");
    List<Example> examples = dataset.getUnlabeledExamples();
    LogInfo.maxIndLevel = 10;
    for(int e = 0; e < examples.size(); e++) {
      if(Execution.shouldBail()) break;
      Example ex = examples.get(e);
      begin_track("Example %d/%d", e, examples.size());
      features.cachePredicates(ex, false);
      learner.labelExample(ex);
      //if(e >= examples.size() / 2) learner.labelExampleSoft(ex);
      out.add(ex);
      end_track();
    }
    out.close();
    end_track();

    if(computeEdgeMutualInfo)
      learner.computeEdgeMutualInfo(dataset.getUnlabeledExamples());

    dataset.clear();
  }

  public static void main(String[] args) {
    Execution.run(args, new Main());
  }
}
