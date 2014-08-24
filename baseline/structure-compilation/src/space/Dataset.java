package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

/**
 * Data format:
 * Each line is a position of a sequence.
 * Each token on the line is the value of a primitive predicate.
 * x_11 x_12 x_13 ... x_1p y_1 |
 * ...                         |-  example 1
 * x_n1 x_n2 x_n3 ... x_np y_n |
 *
 * x_11 x_12 x_13 ... x_1p y_1 |
 * ...                         |-  example 2
 * x_n1 x_n2 x_n3 ... x_np y_n |
 *
 */
class Dataset {
  public static class Options {
    @Option public ArrayList<String> trainPaths = new ArrayList();
    @Option public ArrayList<String> testPaths = new ArrayList();
    @Option public ArrayList<String> unlabeledPaths = new ArrayList();
    @Option public int trainMaxExamples = Integer.MAX_VALUE;
    @Option public int testMaxExamples = Integer.MAX_VALUE;
    @Option public int unlabeledMaxExamples = Integer.MAX_VALUE;
    @Option public int minSequenceLength = 2; // Hack to ignore -DOCSTART-
    @Option public int maxSequenceLength = Integer.MAX_VALUE;
    @Option public boolean serializeExamples = false; // Use serialized list
    @Option public int serializeActiveSize = 10000;
  }

  Options opts;
  List<Example> trainExamples;
  List<Example> testExamples;
  List<Example> unlabeledExamples;

  public Dataset(Options opts) {
    this.opts = opts;
  }

  public List<Example> getTrainExamples() { return trainExamples; }
  public List<Example> getTestExamples() { return testExamples; }
  public List<Example> getUnlabeledExamples() { return unlabeledExamples; }

  public void load() {
    this.trainExamples = new DatasetParser("train", opts.trainPaths, opts.trainMaxExamples).examples;
    this.testExamples = new DatasetParser("test", opts.testPaths, opts.testMaxExamples).examples;
    this.unlabeledExamples = new DatasetParser("unlabeled", opts.unlabeledPaths, opts.unlabeledMaxExamples).examples;
  }

  public void clear() {
    trainExamples.clear();
    testExamples.clear();
    unlabeledExamples.clear();
  }

  class DatasetParser {
    public List<Example> examples;
    private List<String> currSeqLines = new ArrayList();

    public DatasetParser(String prefix, List<String> paths, int maxExamples) {
      if(opts.serializeExamples)
        examples = new SerializedList(opts.serializeActiveSize,
            SerializedList.getTempPath(prefix));
      else
        examples = new ArrayList();
      for(String path : paths)
        if(load(path, maxExamples)) break;
      // For serializedList, flush last block by rewinding (so strings make it out)
      if(examples.size() > 0) examples.get(0);
    }

    // Return whether we've maxed out
    public boolean load(String path, int maxExamples) {
      try {
        begin_track("DatasetParser(%s)", path);
        BufferedReader in = IOUtils.openIn(path);
        String line;
        while((line = in.readLine()) != null) {
          if(line.startsWith("#")) continue;
          if(line.length() == 0) {
            if(flush(maxExamples)) break;
          }
          else
            currSeqLines.add(line);
        }
        in.close();
        flush(maxExamples);
        end_track();
      } catch(IOException e) {
        throw new RuntimeException(e);
      }
      return examples.size() >= maxExamples;
    }

    private boolean flush(int maxExamples) {
      if(examples.size() >= maxExamples) return true;
      if(currSeqLines.size() > 0) {
        if(opts.minSequenceLength <= currSeqLines.size() &&
           opts.maxSequenceLength >= currSeqLines.size()) {
          examples.add(createExample(currSeqLines));
          logs("%d examples", examples.size());
        }
        currSeqLines.clear();
      }
      return false;
    }

    private Example createExample(List<String> lines) {
      int N = lines.size();
      String[][] data = new String[N][];
      for(int i = 0; i < N; i++)
        data[i] = StrUtils.split(lines.get(i));
      return new Example(data);
    }
  }
}
