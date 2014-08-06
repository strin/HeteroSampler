#include "ThreadPool.h"

using namespace std;

int main() {
  auto worker = [] (int tid, int load) {
    cout << "load " << load << endl;
  };
  ThreadPool<int> thread_pool(5, worker);  
  for(int t = 0; t < 10; t++) {
    thread_pool.addWork(t);  
  }
  thread_pool.waitFinish();
  cout << "end " << endl;
  return 0;
}
