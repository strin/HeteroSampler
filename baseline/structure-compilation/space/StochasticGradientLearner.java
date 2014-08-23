package space;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class StochasticGradientLearner extends Learner {
  public enum StepSizeType { constant, diminish, diminish_sqrt };
  public static class Options {
    @Option public double initStepSize = 0.1;
    @Option public double stepSizeIterIncr = 3;
    @Option public StepSizeType stepSizeType = StepSizeType.diminish_sqrt;
  }

  Options sopts;

  public StochasticGradientLearner(Learner.Options opts, Options sopts, Model model) {
    super(opts, model);
    this.sopts = sopts;
  }

  public boolean learnForOneIter() {
    double logLikelihood = 0;
    Performance performance = model.newPerformance();
    int E = trainExamples.size();
    double stepSize = getStepSize(iter);
    logs("stepSize = %s", stepSize);

    begin_track("Examples");
    for(int e = 0; e < E; e++) {
      if(Execution.shouldBail()) break;
      Example ex = trainExamples.get(e);
      begin_track("Example %d/%d", e, E);

      Model.InferState state = model.getInferState(ex, params);
      if(model.opts.piecewise) {
        int pieceSize = model.opts.piecewiseWindowSize*2 + 1;
        for(int i = 0; i < ex.N; i++) {
          int start = state.piecewiseStart(i), end = state.piecewiseEnd(i);
          logLikelihood += state.subInfer(start, end) / E;
          state.subUpdateCounts(stepSize / pieceSize, params, start, end);
        }
      }
      else {
        logLikelihood += state.infer() / E;
        state.updateCounts(stepSize, params);
      }

      // Evaluate the training point
      performance.add(ex.y, state.getBestOutput());

      //model.features.outputParams(Execution.getFile("params."+e), params, 0.01);
      putOutputIterEx(iter, e);
      end_track();
    }
    end_track();

    Execution.putLogRec("trainLogLikelihood", Fmt.D(logLikelihood));
    Execution.putLogRec("trainQuality", Fmt.D(performance.getQuality()));
    performance.log();

    return false;
  }

  public double getStepSize(int iter) {
    double scale;
    switch(sopts.stepSizeType) {
      case constant: scale = 1; break;
      case diminish: scale = iter+sopts.stepSizeIterIncr+1; break;
      case diminish_sqrt: scale = Math.sqrt(iter+sopts.stepSizeIterIncr+1); break;
      default: throw Exceptions.unknownCase;
    }
    return sopts.initStepSize / scale;
  }
}
