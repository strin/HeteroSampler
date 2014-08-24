package space;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class BatchLearner extends Learner {
  public enum MaximizerType { gradient, lbfgs };
  public static class Options {
    @Option public MaximizerType maximizerType = MaximizerType.gradient;
    @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
    @OptionSet(name="lsearch") public BacktrackingLineSearch.Options lsearch = new BacktrackingLineSearch.Options();
  }

  Maximizer maximizer;
  LogLikelihoodFunction func;
  Options bopts;

  public BatchLearner(Learner.Options opts, Options bopts, Model model) {
    super(opts, model);
    this.bopts = bopts;

    switch(bopts.maximizerType) {
      case gradient: this.maximizer = new GradientMaximizer(bopts.lsearch); break;
      case lbfgs: this.maximizer = new LBFGSMaximizer(bopts.lsearch, bopts.lbfgs); break;
      default: throw Exceptions.unknownCase;
    }
  }

  public void setData(List<Example> trainExamples, List<Example> testExamples) {
    super.setData(trainExamples, testExamples);
    this.func = new LogLikelihoodFunction(
        trainExamples, opts.regularization, params);
  }
  
  public void learn() {
    super.learn();
    maximizer.logStats();
    maximizer = null;
  }

  public boolean learnForOneIter() {
    return maximizer.takeStep(func);
  }

  class LogLikelihoodFunction implements FunctionState {
    // Defines the function
    List<Example> examples; // Training examples
    double regularization;

    // Current state
    double[] params;
    double value; // Current value
    double[] gradient; // Gradient
    int valid; // 0 = invalid, 1 = value valid, 2 = value and gradient valid

    public LogLikelihoodFunction(List<Example> examples, double regularization, double[] params) {
      this.examples = examples;
      this.regularization = regularization;
      this.params = params;
      this.gradient = new double[params.length];
    }

    public double[] point() { return params; }

    // Function value at the current point
    public double value() { compute(1); return value; }
    public double[] gradient() { compute(2); return gradient; }
    public void invalidate() { valid = 0; }

    private String validStr() {
      if(valid == 0) return "none";
      if(valid == 1) return "value";
      if(valid == 2) return "gradient";
      throw Exceptions.unknownCase;
    }

    public void compute(int newValid) {
      if(newValid <= valid) return;
      valid = newValid;

      begin_track(String.format("LogLikelihoodFunction.compute(%s): %d examples",
          validStr(), examples.size()), true);

      // Clear gradient
      if(valid >= 1) value = 0;
      if(valid >= 2) ListUtils.set(gradient, 0);

      boolean log = (valid >= 2);

      // Train
      value += new ExampleProcessor("train", trainExamples, true,
         (valid >= 2 ? gradient : null),
         log, log).logLikelihood;

      // Regularize
      double scale = regularization; // / examples.size();
      value -= 0.5 * scale * NumUtils.l2NormSquared(params);
      if(valid >= 2)
        ListUtils.incr(gradient, -scale, params);

      end_track();
    }
  }
}
