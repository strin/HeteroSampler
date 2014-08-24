package space;

import java.io.*;
import java.util.*;
import java.util.regex.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

/**
 * This class allows us to use a large list which resides mostly
 * on disk but is paged into memory as we access points.
 * Best performance if we make sequential passes through the list.
 * If activeSize is large enough, then behavior is exactly like an ArrayList.
 *
 * Remember to call set(i, x) to ensure that i is saved to disk.
 * Not thread-safe!
 */
public class SerializedList<T extends Serializable> extends AbstractList<T> {
  // Input
  private int activeSize; // Number of items to keep in memory
  private String pathPrefix; // Place to store serialized objects

  // State
  private int activeStartIndex; // Start of active block
  private int numItems; // Number of items
  List<T> activeItems;
  boolean dirty;

  public static final int id = new Random().nextInt(100000);

  public static String getTempPath(String prefix) {
    return "/scratch/"+id+"-"+prefix;
  }

  public SerializedList(int activeSize, String pathPrefix) {
    this.activeSize = activeSize;
    this.pathPrefix = pathPrefix;
    this.activeItems = new ArrayList();
  }

  private int block(int i) { return (i / activeSize) * activeSize; }
  private String blockPath(int i) { return pathPrefix+block(i)+".block"; }
  private void loadBlock(int startIndex) {
    startIndex = block(startIndex);
    if(activeStartIndex == startIndex) return; // Already loaded
    flush();
    activeStartIndex = startIndex;
    if(new File(blockPath(activeStartIndex)).exists()) {
      begin_track(String.format("SerializedList.read(%d)", activeStartIndex), true, true);
      StopWatchSet.begin("SerializedList.read");
      activeItems = (List)IOUtils.readObjFileHard(blockPath(activeStartIndex));
      StopWatchSet.end();
      end_track();
    }
    else
      activeItems = new ArrayList();
  }

  public boolean add(T x) {
    loadBlock(numItems);
    activeItems.add(x);
    numItems++;
    dirty = true;
    return true;
  }

  public void clear() {
    for(int i = 0; i < numItems; i += activeSize)
      new File(blockPath(i)).delete();
    activeItems = new ArrayList();
    activeStartIndex = 0;
    numItems = 0;
    dirty = false;
  }

  public T set(int i, T x) {
    loadBlock(i); // Hopefully this doesn't change the active block
    activeItems.set(i-activeStartIndex, x);
    dirty = true;
    return x;
  }
  public void flush() {
    if(dirty) { // Save
      begin_track(String.format("SerializedList.write(%d)", activeStartIndex), true, true);
      StopWatchSet.begin("SerializedList.write");
      IOUtils.writeObjFileHard(blockPath(activeStartIndex), activeItems);
      new File(blockPath(activeStartIndex)).deleteOnExit();
      dirty = false;
      StopWatchSet.end();
      end_track();
    }
  }

  public T get(int i) {
    loadBlock(i);
    return activeItems.get(i-activeStartIndex);
  }
  public int size() { return numItems; }
  public Iterator<T> iterator() { return new SerializedIterator(); }

  public class SerializedIterator implements Iterator<T> {
    private int index;
    public boolean hasNext() { return index < numItems; }
    public T next() {
      loadBlock(index);
      T x = activeItems.get(index-activeStartIndex);
      index++;
      return x;
    }
    public void remove() { throw Exceptions.unsupported; }
  }

  public static void main(String[] args) {
    List<StringBuilder> l = new SerializedList<StringBuilder>(3, ".");
    for(int i = 0; i < 20; i++)
      l.add(new StringBuilder(""+i));
    for(int i = 0; i < l.size(); i++) {
      StringBuilder b = l.get(i);
      b.append('g');
      l.set(i, b);
    }
    System.out.println(l.size());
    for(StringBuilder b : l)
      System.out.println(b);
    l.clear();
  }
}
