package space;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

/**
 * Usage:
 *  - setData()
 *  - learn()
 */
public abstract class Learner {
  public static class Options {
    @Option public int maxIters = 5;
    @Option public int maxThreads = 1;
    @Option public int verbose = 0;
    @Option public double outputParamsThreshold = 1e-4;
    /*@Option public double recordParamsThreshold = 1e-4;
    @Option public int recordParamsInterval = 1;
    @Option public int recordPerformanceInterval = 1;
    @Option public int evalInterval = 1;
    @Option public int recordParamsMax = 10;*/

    @Option public boolean serializeFinalParams = false;
    @Option public double regularization = 0.1;
  }

  Options opts;
  Model model;
  List<Example> trainExamples, testExamples;
  double[] params; // Current parameters
  int iter;

  public Learner(Options opts, Model model) {
    this.opts = opts;
    this.model = model;
    this.params = new double[model.features.size()];
  }

  public void setData(List<Example> trainExamples, List<Example> testExamples) {
    this.trainExamples = trainExamples;
    this.testExamples = testExamples;
  }

  public void learn() {
    begin_track("learn()");
    boolean done = false;
    for(iter = 0; iter < opts.maxIters && !done; iter++) {
      if(Execution.shouldBail()) break;

      begin_track("Iteration %d/%d", iter, opts.maxIters);
      Record.begin("iteration", iter);
      done = learnForOneIter();
      evaluateExamples("test", testExamples);
      Record.end();
      end_track();
    }

    putOutputIterEx(iter, -1);
    recordParams();
    model.features.outputParams(Execution.getFile("params."+iter), params, opts.outputParamsThreshold);
    if(opts.serializeFinalParams) serializeParams(iter);
    end_track();
  }

  public void computeEdgeMutualInfo(List<Example> examples) {
    begin_track("computeEdgeMutualInfo");
    final StatFig miFig = new StatFig();
    final PrintWriter datOut = IOUtils.openOutEasy(Execution.getFile("edgeMutualInfo.dat"));
    final PrintWriter txtOut = IOUtils.openOutEasy(Execution.getFile("edgeMutualInfo.txt"));
    new Parallelizer(opts.maxThreads).process(examples, new Parallelizer.Processor<Example>() {
      public void process(Example ex, int e, int E) {
        boolean log = true;
        if(log) begin_track("Example %d/%d", e, E);
        Model.InferState state = model.getInferState(ex, params);
        state.infer();
        double[] mutualInfos = state.computeEdgeMutualInfo();
        synchronized(miFig) {
          if(txtOut != null) {
            List<String> tokens = new ArrayList();
            for(int i = 0; i < ex.N; i++) {
              String w = ex.data[i][0]; // + "_" + ex.xIndices[i][0]; //(ex.xIndices[i][0] == -1 ? "*" : "");
              if(ex.data[i].length > 1)
                w += "/" +  ex.data[i][ex.data[i].length-1];
              tokens.add(w);
              if(i < ex.N-1) tokens.add(Fmt.D(mutualInfos[i]));
            }
            txtOut.println(StrUtils.join(tokens));
          }
          for(double mi : mutualInfos) {
            miFig.add(mi);
            datOut.println(mi);
          }
        }
        if(log) end_track();
      }
    });
    if(txtOut != null) txtOut.close();
    if(datOut != null) datOut.close();
    logss("Edge mutual information: " + miFig);
    end_track();
  }

  void serializeParams(int iter) {
    try {
      IOUtils.writeObjFile(Execution.getFile("params."+iter), params);
    } catch(IOException e) { 
      throw new RuntimeException(e);
    }
  }

  protected void putOutputIterEx(int iter, int ex) {
    Execution.putOutput("currIter", iter);
    Execution.putOutput("currExample", ex);
  }

  void recordParams() {
    StopWatchSet.begin("recordParams");
    StopWatchSet.end();
  }

  // Return whether we're done
  public abstract boolean learnForOneIter();

  public void evaluateExamples(String prefix, List<Example> examples) {
    // Evaluate test examples
    begin_track(String.format("Evaluating on %s: %d examples", prefix, examples.size()), true);
    new ExampleProcessor(prefix, examples, true, null, true, true);
    end_track();
  }

  public void labelExamples(final List<Example> examples) {
    begin_track("Label examples");
    new Parallelizer(opts.maxThreads).process(examples, new Parallelizer.Processor<Example>() {
      public void process(Example ex, int i, int n) {
        boolean log = true;
        if(log) begin_track("Example %d/%d", i, n);
        labelExample(ex);
        examples.set(i, ex); // if SerializedList
        if(log) end_track();
      }
    });
    end_track();
  }
  public void labelExample(Example ex) {
    Model.InferState state = model.getInferState(ex, params);
    ex.y = state.getBestOutput();
  }
  public void labelExampleSoft(Example ex) {
    Model.InferState state = model.getInferState(ex, params);
    state.infer();
    int T = model.T;
    double[] nodePosteriors = new double[T];
    for(int i = 0; i < ex.N; i++) {
      state.FB.getNodePosteriors(i, nodePosteriors);
      //dbg(ex.data[i][0] + " " + Fmt.D(nodePosteriors));
    }
  }

  class ExampleProcessor {
    // Output
    double logLikelihood;
    Performance performance;

    public ExampleProcessor(String prefix, List<Example> examples,
        final boolean computeLogLikelihood, final double[] counts,
        final boolean evaluatePerformance,
        boolean log) {
      if(evaluatePerformance)
        this.performance = model.newPerformance();

      begin_track("Examples");
      new Parallelizer(opts.maxThreads).process(examples, new Parallelizer.Processor<Example>() {
        public void process(Example ex, int i, int n) {
          boolean log = true;
          if(log) begin_track("Example %d/%d", i, n);
          Model.InferState state = model.getInferState(ex, params);
          if(computeLogLikelihood) {
            double exLogLikelihood = state.infer();
            synchronized(this) { logLikelihood += exLogLikelihood / n; }
            if(counts != null) {
              synchronized(this) { state.updateCounts(1.0/n, counts); }
            }
          }
          if(evaluatePerformance)
            performance.add(ex.y, state.getBestOutput());
          if(log) putOutputIterEx(iter, i);
          if(log) end_track();
        }
      });
      end_track();

      if(computeLogLikelihood && log)
        Execution.putLogRec(prefix+"LogLikelihood", Fmt.D(logLikelihood));
      if(evaluatePerformance && log) {
        Execution.putLogRec(prefix+"Quality", Fmt.D(performance.getQuality()));
        performance.log();
      }
    }
  }
}
