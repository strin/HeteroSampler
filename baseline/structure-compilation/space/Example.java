package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import static fig.basic.LogInfo.*;

/**
 * An example is a sequence.
 */
public class Example implements java.io.Serializable {
  // Raw data
  String[][] data; // position i -> tokens at that point
  int N; // Length of sequence

  // Cached values based on predicates
  int[][] xIndices; // position i -> input indices of predicates
  int[] y; // position i -> output label at position i

  int[][] sigs; // For OCR

  public Example(String[][] data) {
    this.data = data;
    this.N = data.length;
  }

  // If this flag is set, don't write data out any more
  public transient static boolean serializeStringData = true;

  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    boolean haveStringData = in.readBoolean();
    if(haveStringData) data = (String[][])in.readObject();
    this.N = in.readInt();

    if(in.readBoolean()) {
      int P = in.readInt();
      this.xIndices = new int[N][P];
      for(int i = 0; i < N; i++)
        for(int p = 0; p < P; p++)
          xIndices[i][p] = in.readInt();
      this.y = new int[N];
      for(int i = 0; i < N; i++)
        y[i] = in.readInt();
    }
  }
  private void writeObject(ObjectOutputStream out) throws IOException {
    out.writeBoolean(serializeStringData);
    if(serializeStringData) out.writeObject(data);
    out.writeInt(N);

    if(xIndices != null) {
      out.writeBoolean(true);
      int P = xIndices[0].length;
      out.writeInt(P);
      for(int i = 0; i < N; i++)
        for(int p = 0; p < P; p++)
          out.writeInt(xIndices[i][p]);
      for(int i = 0; i < N; i++)
        out.writeInt(y[i]);
    }
    else
      out.writeBoolean(false);
  }
}

interface ExampleProcessor {
  public void process(Example ex);
}

class ExampleWriter {
  boolean first;
  PrintWriter out;
  Indexer<String> tagIndexer;

  public ExampleWriter(String path, Indexer<String> tagIndexer) {
    this.out = IOUtils.openOutEasy(path);
    this.tagIndexer = tagIndexer;
    this.first = true;
  }

  public void add(Example ex) {
    if(out == null) return;
    if(!first) out.println("");
    first = false;
    for(int i = 0; i < ex.N; i++) {
      if(ex.data == null)
        out.println(tagIndexer.getObject(ex.y[i]));
      else
        out.println(StrUtils.join(ex.data[i], " ") + " " + tagIndexer.getObject(ex.y[i]));
    }
  }

  public void addAll(List<Example> examples) {
    for(Example ex : examples) add(ex);
  }

  public void close() { if(out != null) out.close(); }

  public static void write(String path, Indexer<String> tagIndexer, List<Example> examples) {
    ExampleWriter out = new ExampleWriter(path, tagIndexer);
    out.addAll(examples);
    out.close();
  }
}
