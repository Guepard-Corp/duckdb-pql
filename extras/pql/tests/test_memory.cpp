// Memory discipline tests: nothing on a hot path reaches the allocator, the
// arena is sized exactly, and a trained model keeps none of its data.
//
//   make -C extras/pql test_memory
//
// The allocation counts come from a global operator new, so this file must not
// be linked with anything that also replaces it.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#ifdef __APPLE__
#include <malloc/malloc.h>
static size_t UsableSize(void *p) { return malloc_size(p); }
#else
#include <malloc.h>
static size_t UsableSize(void *p) { return malloc_usable_size(p); }
#endif
static std::atomic<unsigned long long> g_allocs{0};
static std::atomic<long long> g_live{0};
void *operator new(std::size_t n) {
  void *p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  g_allocs++; g_live += (long long)UsableSize(p);
  return p;
}
void *operator new[](std::size_t n) { return operator new(n); }
void operator delete(void *p) noexcept { if (p) { g_live -= (long long)UsableSize(p); std::free(p); } }
void operator delete[](void *p) noexcept { operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { operator delete(p); }

#include "pql.hpp"
#include <random>
using namespace pql;
static const double DAY = 86400.0 * 1e6;
static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { if (fails < 20) { printf("   FAIL: " __VA_ARGS__); printf("\n"); } fails++; } } while (0)

static Column MakeCol(const char *name, ColType t, size_t n) {
  Column c; c.name = name; c.type = t;
  c.num.assign(n, 0.0); c.code.assign(n, 0u); c.valid.assign(n, 1);
  return c;
}
// users -> orders -> items, so a two-hop SAGE model has a grandchild relation.
static Database MakeDb(uint32_t seed, size_t n_users) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0, 1);
  Database db;
  Frame users; users.name = "users";
  Column uid = MakeCol("id", ColType::INT64, n_users), uts = MakeCol("ts", ColType::TIMESTAMP, n_users);
  Column ten = MakeCol("tenure", ColType::DOUBLE, n_users), reg = MakeCol("region", ColType::CATEGORY, n_users);
  for (const char *l : {"a", "b", "c", "d", "e"}) reg.dict.Intern(l);
  std::vector<double> rate(n_users), anchor(n_users);
  for (size_t i = 0; i < n_users; i++) {
    uid.num[i] = double(i); anchor[i] = (500.0 + U(rng) * 500.0) * DAY; uts.num[i] = anchor[i];
    rate[i] = U(rng); ten.num[i] = 100.0 + U(rng) * 900.0; reg.code[i] = (uint32_t)(i % 5) + 1u;
  }
  users.columns = {uid, uts, ten, reg}; users.nrows = n_users;
  std::vector<double> o_uid, o_ts, o_amt;
  for (size_t i = 0; i < n_users; i++) {
    const int n_past = int(rate[i] * 25.0);
    for (int k = 0; k < n_past; k++) { o_uid.push_back(double(i)); o_ts.push_back(anchor[i] - U(rng) * 300.0 * DAY); o_amt.push_back(10.0 + U(rng) * 5.0); }
    if (U(rng) < rate[i]) { o_uid.push_back(double(i)); o_ts.push_back(anchor[i] + (1.0 + U(rng) * 25.0) * DAY); o_amt.push_back(10.0 + U(rng) * 5.0); }
  }
  const size_t no = o_uid.size();
  Frame ord; ord.name = "orders";
  Column oi = MakeCol("id", ColType::INT64, no), ou = MakeCol("user_id", ColType::INT64, no);
  Column ot = MakeCol("ts", ColType::TIMESTAMP, no), oa = MakeCol("amount", ColType::DOUBLE, no);
  for (size_t i = 0; i < no; i++) { oi.num[i] = double(i); ou.num[i] = o_uid[i]; ot.num[i] = o_ts[i]; oa.num[i] = o_amt[i]; }
  ord.columns = {oi, ou, ot, oa}; ord.nrows = no;
  const size_t ni = no * 3;
  Frame it; it.name = "items";
  Column ii = MakeCol("id", ColType::INT64, ni), io = MakeCol("order_id", ColType::INT64, ni), iq = MakeCol("qty", ColType::DOUBLE, ni);
  for (size_t i = 0; i < ni; i++) { ii.num[i] = double(i); io.num[i] = double(i / 3); iq.num[i] = 1.0 + U(rng) * 4.0; }
  it.columns = {ii, io, iq}; it.nrows = ni;
  db.tables.push_back(users); db.tables.push_back(ord); db.tables.push_back(it);
  db.fks.push_back({"orders", "user_id", "users", "id"});
  db.fks.push_back({"items", "order_id", "orders", "id"});
  db.BuildLinks({});
  return db;
}

static Statement TrainStmt(const std::string &opts) {
  return Parse("TRAIN MODEL m PREDICT EXISTS(orders) FOR users AT ts HORIZON 30 DAYS OPTIONS (" + opts + ")");
}

// Allocations per epoch after the first, as seen from the per-epoch callback.
// Everything the first epoch sets up is excluded; what remains is the steady
// state.
static std::vector<unsigned long long> EpochAllocs(const Database &db, const std::string &opts, Model &m) {
  std::vector<unsigned long long> out;
  unsigned long long last = 0; bool first = true;
  auto cb = [&]() { const unsigned long long now = g_allocs.load(); if (!first) out.push_back(now - last); first = false; last = now; };
  TrainModel(db, TrainStmt(opts), m, cb);
  return out;
}

int main() {
  {
    printf("== A. zero heap allocations per training step\n");
    // A step-level allocation would scale with the number of steps, so the
    // per-epoch count at batch 8 (8x the steps of batch 64) must equal the
    // count at batch 64. What is left is validation bookkeeping, and is bounded.
    Database db = MakeDb(7, 600);
    const char *archs[] = {"", ", arch = 'sage'", ", arch = 'sage', layers = 2"};
    for (const char *arch : archs) {
      Model m64, m8;
      auto e64 = EpochAllocs(db, std::string("epochs = 5, hidden = 32, batch = 64") + arch, m64);
      auto e8 = EpochAllocs(db, std::string("epochs = 5, hidden = 32, batch = 8") + arch, m8);
      unsigned long long mx64 = 0, mx8 = 0;
      for (auto v : e64) mx64 = std::max(mx64, v);
      for (auto v : e8) mx8 = std::max(mx8, v);
      printf("   arch '%s': per-epoch allocs batch64 max=%llu batch8 max=%llu\n", *arch ? arch + 2 : "mlp", mx64, mx8);
      CHECK(mx64 == mx8, "per-epoch allocations depend on the step count => something allocates per step");
      CHECK(mx64 <= 12, "per-epoch allocations should be a handful of validation vectors, got %llu", mx64);
    }
  }
  {
    printf("== B. prediction allocates per call, not per row\n");
    Database db = MakeDb(7, 600);
    for (const char *arch : {"", ", arch = 'sage'"}) {
      Model m; TrainModel(db, TrainStmt(std::string("epochs = 2, hidden = 16") + arch), m);
      auto all = Parse("PREDICT EXISTS(orders) FOR users USING MODEL m");
      auto few = Parse("PREDICT EXISTS(orders) FOR users WHERE id < 5 USING MODEL m");
      unsigned long long a0 = g_allocs.load(); auto p_all = RunPredict(db, m, all); unsigned long long a1 = g_allocs.load();
      auto p_few = RunPredict(db, m, few); unsigned long long a2 = g_allocs.load();
      printf("   arch '%s': %zu rows -> %llu allocs, %zu rows -> %llu allocs\n", *arch ? "sage" : "mlp", p_all.size(), a1 - a0, p_few.size(), a2 - a1);
      CHECK(p_all.size() > 100 && p_few.size() == 5, "unexpected prediction counts");
      CHECK((a1 - a0) <= (a2 - a1) + 8, "predicting %zu rows allocated %llu more times than predicting 5", p_all.size(), (a1 - a0) - (a2 - a1));
    }
  }
  {
    printf("== C. arena: counting mode sizes the block exactly\n");
    std::mt19937 rng(3);
    double worst = 0;
    for (int trial = 0; trial < 2000; trial++) {
      const int k = 1 + int(rng() % 12);
      std::vector<size_t> ns(k);
      for (auto &n : ns) n = rng() % 5 == 0 ? 0 : rng() % 3000;
      std::vector<void *> ptrs;
      auto carve = [&](Arena &a) {
        ptrs.clear();
        for (int i = 0; i < k; i++) {
          switch (i % 3) {
          case 0: ptrs.push_back(a.Zeroed<float>(ns[i])); break;
          case 1: ptrs.push_back(a.Alloc<double>(ns[i])); break;
          default: ptrs.push_back(a.Alloc<uint32_t>(ns[i])); break;
          }
        }
      };
      const size_t need = Arena::Measure(carve);
      CHECK(ptrs.size() == (size_t)k, "measure ran the carve");
      for (void *p : ptrs) CHECK(p == nullptr, "counting mode must hand out no memory");
      Arena a; a.Carve(carve);
      CHECK(a.Capacity() == need && a.Used() == need, "Carve: used %zu cap %zu need %zu", a.Used(), a.Capacity(), need);
      worst = std::max(worst, double(a.Used()) / double(std::max<size_t>(1, a.Capacity())));
      // Every pointer 16-aligned, inside the block, and no two spans overlap.
      std::vector<std::pair<uintptr_t, uintptr_t>> spans;
      for (int i = 0; i < k; i++) {
        const uintptr_t b = (uintptr_t)ptrs[i];
        const size_t sz = ns[i] * (i % 3 == 0 ? 4 : i % 3 == 1 ? 8 : 4);
        CHECK(b % 16 == 0, "pointer not 16-aligned");
        for (auto &s : spans) CHECK(b + sz <= s.first || s.second <= b, "spans overlap");
        spans.emplace_back(b, b + sz);
      }
      // One byte past the plan is a clean error, not a stomp.
      bool threw = false;
      try { a.Alloc<uint8_t>(1); } catch (const std::runtime_error &) { threw = true; }
      CHECK(threw, "allocation past the plan must throw");
      // Alloc<T>(0) never fails, even on an empty arena.
      Arena e; e.Reserve(0); e.Alloc<float>(0); e.Zeroed<double>(0);
    }
    printf("   2000 random carves: max Used/Capacity = %.4f (exact by construction)\n", worst);
    bool threw = false;
    try { Arena a; a.Reserve(64); a.Alloc<double>(std::numeric_limits<size_t>::max() / 4); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw, "n*sizeof(T) overflow must throw, not wrap");
    threw = false;
    try { Arena::Measure([](Arena &a) { a.Alloc<uint64_t>(std::numeric_limits<size_t>::max() / 2); }); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw, "overflow in counting mode must throw too");
  }
  {
    printf("== D. SageScratch and TrainModel never outgrow their arena\n");
    std::mt19937 rng(11);
    for (int trial = 0; trial < 300; trial++) {
      SageParams p;
      p.channels = 1 + int(rng() % 130);
      p.self_dim = int(rng() % 40);
      const size_t E = rng() % 4;
      p.w_self.assign((size_t)p.channels * (size_t)std::max(1, p.self_dim), 0.0f);
      p.bias.assign((size_t)p.channels, 0.0f);
      p.w_out.assign((size_t)p.channels, 0.0f);
      for (size_t e = 0; e < E; e++) { p.src_dim.push_back(int(rng() % 70)); p.w_neigh.emplace_back((size_t)p.channels * (size_t)std::max(1, p.src_dim.back()), 0.0f); }
      const size_t B = 1 + rng() % 700;
      SageScratch sc; sc.Ensure(B, p);
      CHECK(sc.arena.Used() == sc.arena.Capacity(), "SageScratch: used %zu != cap %zu", sc.arena.Used(), sc.arena.Capacity());
      CHECK(sc.xself && sc.z && sc.h && sc.dz && sc.obuf, "SageScratch pointers must be real");
      for (float *mp : sc.means) CHECK(mp != nullptr, "means pointer null");
      // Re-Ensure with a smaller shape keeps the block; a larger one grows it.
      sc.Ensure(B / 2 + 1, p);
      CHECK(sc.arena.Used() <= sc.arena.Capacity(), "re-Ensure smaller");
    }
    // The whole training path across shapes: a wrong plan throws
    // "arena exhausted", which is what this guards against.
    Database db = MakeDb(5, 300);
    const int hiddens[] = {1, 3, 17, 64, 130};
    const int batches[] = {1, 7, 64, 1000};
    int runs = 0;
    for (int h : hiddens) for (int b : batches) for (const char *arch : {"", ", arch = 'sage'", ", arch = 'sage', layers = 2"}) {
      if (h == 130 && *arch) continue; // slow, and the scratch path is shape-agnostic
      char opts[160]; snprintf(opts, sizeof opts, "epochs = 2, hidden = %d, batch = %d%s", h, b, arch);
      Model m;
      try { TrainModel(db, TrainStmt(opts), m); runs++; }
      catch (const std::exception &ex) { CHECK(false, "TrainModel(%s) threw: %s", opts, ex.what()); }
    }
    printf("   %d shape combinations trained without an arena error\n", runs);
  }
  {
    printf("== E. a trained model keeps none of the training data\n");
    const long long before = g_live.load();
    Model m; size_t data_bytes = 0;
    {
      Database db = MakeDb(9, 2000);
      for (auto &t : db.tables) for (auto &c : t.columns) data_bytes += c.num.size() * 8 + c.code.size() * 4 + c.valid.size();
      TrainModel(db, TrainStmt("epochs = 3, hidden = 64"), m);
    }
    const long long kept = g_live.load() - before;
    // TransposeScratch is a thread-local that stays for the process; it is
    // bounded by the largest weight matrix, which is part of ApproxBytes.
    const long long allowed = (long long)m.ApproxBytes() * 4 + 65536;
    printf("   database %zu bytes; live after it is gone: %lld bytes (model ApproxBytes %zu)\n", data_bytes, kept, m.ApproxBytes());
    CHECK(kept < allowed, "model retains %lld bytes, more than %lld", kept, allowed);
  }
  if (fails) { printf("\n%d memory test(s) FAILED\n", fails); return 1; }
  printf("\nall memory tests passed\n");
  return 0;
}
