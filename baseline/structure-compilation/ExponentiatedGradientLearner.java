package space;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

/**
 * Doesn't work on a simple example.  Bug somewhere!
 */
public class ExponentiatedGradientLearner extends Learner {
  public static class Options {
    @Option public double stepSize = 0.1;
    @Option public int serializeActiveSize = 10000;
  }

  double stepSize;
  int E; // Number of training examples
  double paramsScale;
  List<ExamplePotentials> potentialsList;
  Options egopts;

  public ExponentiatedGradientLearner(Learner.Options opts, Options egopts, Model model) {
    super(opts, model);
    this.egopts = egopts;
    this.stepSize = egopts.stepSize;
  }

  @Override public void setData(List<Example> trainExamples, List<Example> testExamples) {
    super.setData(trainExamples, testExamples);
    this.E = trainExamples.size();
    this.paramsScale = 1.0/(opts.regularization*E);
    this.potentialsList = new SerializedList<ExamplePotentials>(egopts.serializeActiveSize,
        SerializedList.getTempPath("eg"));
  }

  @Override public void learn() {
    initParams();
    super.learn();
  }

  private void initParams() {
    // Initialize alpha distributions over examples to uniform
    // and compute the parameter vector (params)
    begin_track("Init parameters");
    ListUtils.set(params, 0);
    for(int e = 0; e < E; e++) {
      begin_track("Example %d/%d", e, E);
      Example ex = trainExamples.get(e);
      Model.InferState state = model.getInferState(ex, params);
      state.updateCountsNumerator(+paramsScale, params);
      state.updateCountsUniform(-paramsScale, params);
      end_track();
    }
    end_track();
  }

  private double updatePotential(double potential, double score) {
    dbgs("%s %s %s", potential, score,
        Math.pow(potential, 1-stepSize) *
        Math.exp(stepSize * score / (opts.regularization * E)));
    if(potential != -1)
      return (1-stepSize)*Math.log(potential) + stepSize * score * paramsScale;

    double newPotential =
        Math.pow(potential, 1-stepSize) *
        Math.exp(stepSize * score / (opts.regularization * E));
    assert NumUtils.isFinite(newPotential) : potential + " " + score + " " + newPotential;
    return newPotential;
  }

  private void updatePotentials(Example ex, ExamplePotentials potentials, double[] params) {
    // \alpha_{i,y} <- \alpha_{i,y} exp(-\eta \nabla_{i,y})
    int N = potentials.nodePotentials.length;
    int T = potentials.nodePotentials[0].length;
    if(potentials.edgePotentials != null) {
      for(int s = 0; s < T; s++)
        for(int t = 0; t < T; t++)
          potentials.edgePotentials[s][t] =
            updatePotential(potentials.edgePotentials[s][t],
                model.features.edgeScore(s, t, params));
      NumUtils.expNormalize(potentials.edgePotentials);
    }
    //dbg("NODE");

    for(int i = 0; i < N; i++) {
      for(int t = 0; t < T; t++)
        potentials.nodePotentials[i][t] =
          updatePotential(potentials.nodePotentials[i][t],
            model.features.nodeScore(ex, i, t, params));
      NumUtils.expNormalize(potentials.nodePotentials[i]);
    }

    potentials.normalizeRange();
  }

  public boolean learnForOneIter() {
    //double logLikelihood = 0;
    Performance performance = model.newPerformance();

    begin_track("Examples");
    for(int e = 0; e < E; e++) {
      begin_track("Example %d/%d", e, E);
      Example ex = trainExamples.get(e);

      // Initialize potentials
      if(e == potentialsList.size())
        potentialsList.add(new ExamplePotentials(ex, model.features, model.opts.includeEdges));

      // Infer twice to get marginals
      Model.InferState state = model.getInferState(ex, params);
      state.initFB(potentialsList.get(e));
      state.FB.infer(); // Old
      state.FB.saveState();
      updatePotentials(ex, potentialsList.get(e), params);
      potentialsList.set(e, potentialsList.get(e));
      state.FB.infer(); // New

      dbg("PPP " + Fmt.D(params));

      // Add new expected sufficient statistics
      state.updateCountsDenominator(-paramsScale, params);
      // Remove old expected sufficient statistics
      state.FB.restoreState();
      state.updateCountsDenominator(+paramsScale, params);

      dbg("PPP " + Fmt.D(params));

      // Evaluate the training point (with new parameters)
      performance.add(ex.y, state.getBestOutput());

      //model.features.outputParams(Execution.getFile("params."+e), params, 0.01);
      putOutputIterEx(iter, e);
      end_track();
    }
    end_track();

    //Execution.putLogRec("trainLogLikelihood", Fmt.D(logLikelihood));
    Execution.putLogRec("trainQuality", Fmt.D(performance.getQuality()));
    performance.log();
    evaluateExamples("train", trainExamples);

    return false;
  }
}
