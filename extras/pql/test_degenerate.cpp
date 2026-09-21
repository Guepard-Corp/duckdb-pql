// Databases that should not exist but do. Empty tables, one row, every value
// identical, duplicate keys, orphaned children, NaN, infinity, values at the
// edge of double, timestamps at absurd extremes.
//
// Each must either train and give a finite answer, or refuse with a message
// beginning "pql:". Crashing, hanging, and returning a number that is not one
// are all failures, and the third is the worst: a NaN in a column used to make
// every weight NaN and still report a metric of 0.000000, which reads as a
// perfect model.
//
// Run it under sanitizers:
//   clang++ -std=c++17 -O1 -g -I. -fsanitize=address,undefined -o /tmp/d test_degenerate.cpp
#include "src/pql.hpp"
#include <cmath>
#include <cstdio>
#include <string>
using namespace pql;
static const double DAY = 86400.0 * 1e6;
static int fails = 0;
static Column MC(const char *n, ColType t, size_t s) {
  Column c; c.name=n; c.type=t; c.num.assign(s,0.0); c.code.assign(s,0u); c.valid.assign(s,1);
  return c;
}
struct Rows { std::vector<double> uid, uts, ux; std::vector<double> eid, euid, ets, eamt; };
static Database Make(const Rows &r) {
  Database db; Frame u; u.name="users";
  const size_t NU=r.uid.size();
  Column a=MC("id",ColType::INT64,NU), b=MC("ts",ColType::TIMESTAMP,NU), c=MC("x",ColType::DOUBLE,NU);
  for(size_t i=0;i<NU;i++){a.num[i]=r.uid[i];b.num[i]=r.uts[i];c.num[i]=r.ux[i];}
  u.columns={a,b,c}; u.nrows=NU;
  Frame e; e.name="events"; const size_t NE=r.eid.size();
  Column p=MC("id",ColType::INT64,NE), q=MC("user_id",ColType::INT64,NE),
         s=MC("ts",ColType::TIMESTAMP,NE), t=MC("amount",ColType::DOUBLE,NE);
  for(size_t i=0;i<NE;i++){p.num[i]=r.eid[i];q.num[i]=r.euid[i];s.num[i]=r.ets[i];t.num[i]=r.eamt[i];}
  e.columns={p,q,s,t}; e.nrows=NE;
  db.tables={u,e}; db.fks.push_back({"events","user_id","users","id"});
  db.BuildLinks({}); return db;
}
static void Try(const char *name, const Database &db, const char *q) {
  try {
    auto st = Parse(q);
    Model m; TrainReport r = TrainModel(db, st, m);
    const bool bad = std::isinf(r.test_metric);
    printf("  %-34s trained: %s=%.4f rows=%zu%s\n", name, r.metric_name.c_str(),
           r.test_metric, r.n_train, bad ? "   <-- INFINITE METRIC" : "");
    if (bad) fails++;
    // prediction must survive too
    auto ps = Parse((std::string("PREDICT ") + (st.target.ToString()) +
                     " FOR users USING MODEL " + st.model).c_str());
    auto preds = RunPredict(db, m, ps);
    for (const auto &p : preds) {
      if (std::isnan(p.value) || std::isinf(p.value)) {
        printf("      PREDICT produced %f\n", p.value); fails++; break;
      }
    }
  } catch (const std::exception &e) {
    std::string msg = e.what();
    printf("  %-34s refused: %s\n", name, msg.substr(0, 62).c_str());
    if (msg.find("pql:") != 0) { printf("      (message does not start with 'pql:')\n"); fails++; }
  }
}
int main() {
  const char *Q = "TRAIN MODEL m PREDICT COUNT(events) FOR users AT ts HORIZON 30 DAYS "
                  "OPTIONS (epochs = 5, hidden = 8)";
  { Rows r; Try("empty entity table", Make(r), Q); }
  { Rows r; r.uid={0}; r.uts={1000*DAY}; r.ux={1.0}; Try("one entity row, no events", Make(r), Q); }
  { Rows r; for(int i=0;i<50;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);}
    Try("50 rows, empty child table", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back(1000*DAY);r.ux.push_back(7.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back(999*DAY);r.eamt.push_back(3.0);}
    Try("every value identical", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);
      r.ux.push_back(i%2?1e308:-1e308);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(1e308);}
    Try("values at the edge of double", Make(r), Q); }
  { Rows r; const double inf=std::numeric_limits<double>::infinity();
    for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);
      r.ux.push_back(i%3==0?inf:(i%3==1?-inf:0.0));
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(0.0);}
    Try("infinities in a feature column", Make(r), Q); }
  { Rows r; const double nan=std::numeric_limits<double>::quiet_NaN();
    for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);
      r.ux.push_back(i%2?nan:1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(nan);}
    Try("NaN in a feature column", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(7);  // every entity has the same key
      r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(7);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(1.0);}
    Try("duplicate entity keys", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);}
    for(int i=0;i<2000;i++){r.eid.push_back(i);r.euid.push_back(99999);  // no parent matches
      r.ets.push_back(999*DAY);r.eamt.push_back(1.0);}
    Try("child rows with no parent", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back(0.0);r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back(0.0);r.eamt.push_back(1.0);}
    Try("every timestamp is the epoch", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);
      r.uts.push_back(i%2 ? 1e18 : -1e18);   // timestamps at absurd extremes
      r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back(i%2?1e18:-1e18);r.eamt.push_back(1.0);}
    Try("timestamps at +/- 1e18", Make(r), Q); }
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((3000.0+i)*DAY);r.eamt.push_back(1.0);}
    Try("every event is after every anchor", Make(r), Q); }
  // ---- a classifier with one class must refuse, not train a constant --------
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(1.0);}
    Database db = Make(r);
    bool refused = false; std::string msg;
    try { auto st = Parse("TRAIN MODEL m PREDICT EXISTS(events) FOR users AT ts HORIZON 30 DAYS OPTIONS (epochs = 3, hidden = 8)");
          Model m; TrainModel(db, st, m); }
    catch (const std::exception &e) { refused = true; msg = e.what(); }
    printf("  %-34s %s\n", "EXISTS with no positive anywhere", refused ? msg.substr(0, 62).c_str() : "TRAINED (wrong)");
    if (!refused || msg.find("no positives") == std::string::npos) { printf("      expected a refusal naming the missing class\n"); fails++; }
  }
  // ---- empty folds are undefined, not 0.0 -----------------------------------
  { Rows r; for(int i=0;i<5;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(double(i%2));}
    Database db = Make(r);
    auto st = Parse("TRAIN MODEL a PREDICT users.x FOR users OPTIONS (epochs = 3, hidden = 8)");
    Model m; TrainReport rep = TrainModel(db, st, m);
    printf("  %-34s train=%zu val=%zu test=%zu val=%f test=%f\n", "5 rows: no validation or test fold", rep.n_train, rep.n_val, rep.n_test, rep.val_metric, rep.test_metric);
    if (rep.n_val != 0 || rep.n_test != 0 || !std::isnan(rep.val_metric) || !std::isnan(rep.test_metric)) {
      printf("      a fold that was never scored must report NaN, not a number\n"); fails++;
    }
  }
  // ---- cancellation fires between minibatches, not only between epochs ------
  { Rows r; for(int i=0;i<400;i++){r.uid.push_back(i);r.uts.push_back((1000.0+i)*DAY);r.ux.push_back(1.0);
      r.eid.push_back(i);r.euid.push_back(i);r.ets.push_back((999.0+i)*DAY);r.eamt.push_back(1.0);}
    Database db = Make(r);
    int calls = 0; bool thrown = false;
    try { auto st = Parse("TRAIN MODEL c PREDICT COUNT(events) FOR users AT ts HORIZON 30 DAYS OPTIONS (epochs = 5, hidden = 8, batch = 64)");
          Model m; TrainModel(db, st, m, [&]() { if (++calls == 3) throw std::runtime_error("pql: cancelled"); }); }
    catch (const std::exception &) { thrown = true; }
    printf("  %-34s cancel callback fired %d times before the throw\n", "cancel mid-epoch", calls);
    // 240 training rows at batch 64 is four minibatches per epoch, so a third
    // call that stops the run proves the check runs inside the epoch.
    if (!thrown || calls != 3) { printf("      the cancel check must run once per minibatch\n"); fails++; }
    auto st2 = Parse("TRAIN MODEL c2 PREDICT COUNT(events) FOR users AT ts HORIZON 30 DAYS OPTIONS (epochs = 2, hidden = 8)");
    Model m2; TrainModel(db, st2, m2);
    int pcalls = 0; bool pthrown = false;
    try { RunPredict(db, m2, Parse("PREDICT COUNT(events) FOR users USING MODEL c2"), [&]() { if (++pcalls == 1) throw std::runtime_error("pql: cancelled"); }); }
    catch (const std::exception &) { pthrown = true; }
    if (!pthrown) { printf("      PREDICT ignored its cancel callback\n"); fails++; }
  }
  printf("\n%s\n", fails ? "FAILURES" : "every degenerate case handled");
  return fails ? 1 : 0;
}
