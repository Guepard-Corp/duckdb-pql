#include "src/pql.hpp"
#include <cstdio>
#include <random>
using namespace pql;
static const double DAY = 86400.0 * 1e6;

static Column MakeCol(const char *name, ColType t, size_t n) {
  Column c; c.name = name; c.type = t;
  c.num.assign(n, 0.0); c.code.assign(n, 0u); c.valid.assign(n, 1);
  return c;
}

// Build users + events. `couple` controls whether the future depends on the past;
// with couple=false the future is independent, so any AUROC above chance means a leak.
static Database MakeDb(bool couple, uint32_t seed, size_t n_users = 600) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0, 1);
  Database db;
  Frame users; users.name = "users";
  Column uid = MakeCol("id", ColType::INT64, n_users);
  Column uts = MakeCol("ts", ColType::TIMESTAMP, n_users);
  Column ten = MakeCol("tenure", ColType::DOUBLE, n_users);
  std::vector<double> rate(n_users), anchor(n_users);
  for (size_t i = 0; i < n_users; i++) {
    uid.num[i] = double(i);
    anchor[i] = (500.0 + U(rng) * 500.0) * DAY;
    uts.num[i] = anchor[i];
    rate[i] = U(rng);
    ten.num[i] = 100.0 + U(rng) * 900.0;
  }
  users.columns = {uid, uts, ten}; users.nrows = n_users;

  std::vector<double> e_uid, e_ts, e_amt;
  for (size_t i = 0; i < n_users; i++) {
    int n_past = int(rate[i] * 25.0);
    for (int k = 0; k < n_past; k++) {
      e_uid.push_back(double(i));
      e_ts.push_back(anchor[i] - U(rng) * 300.0 * DAY);
      e_amt.push_back(10.0 + U(rng) * 5.0);
    }
    double p_future = couple ? rate[i] : U(rng);
    if (U(rng) < p_future) {
      e_uid.push_back(double(i));
      e_ts.push_back(anchor[i] + (1.0 + U(rng) * 25.0) * DAY);
      e_amt.push_back(10.0 + U(rng) * 5.0);
    }
  }
  const size_t ne = e_uid.size();
  Frame ev; ev.name = "events";
  Column ei = MakeCol("id", ColType::INT64, ne);
  Column eu = MakeCol("user_id", ColType::INT64, ne);
  Column et = MakeCol("ts", ColType::TIMESTAMP, ne);
  Column ea = MakeCol("amount", ColType::DOUBLE, ne);
  for (size_t i = 0; i < ne; i++) { ei.num[i]=double(i); eu.num[i]=e_uid[i]; et.num[i]=e_ts[i]; ea.num[i]=e_amt[i]; }
  ev.columns = {ei, eu, et, ea}; ev.nrows = ne;

  db.tables.push_back(users); db.tables.push_back(ev);
  db.fks.push_back({"events", "user_id", "users", "id"});
  db.BuildLinks({});
  return db;
}

int main() {
  int fails = 0;
  {
    printf("== A. learnable signal (future depends on past)\n");
    Database db = MakeDb(true, 7);
    auto st = Parse("TRAIN MODEL m PREDICT EXISTS(events) FOR users AT ts HORIZON 30 DAYS "
                    "OPTIONS (epochs = 120, hidden = 32, lr = 0.02)");
    Model m; auto rep = TrainModel(db, st, m);
    printf("   width=%d train=%zu val=%zu test=%zu val_%s=%.4f test=%.4f\n", rep.width, rep.n_train,
           rep.n_val, rep.n_test, rep.metric_name.c_str(), rep.val_metric, rep.test_metric);
    if (rep.test_metric < 0.70) { printf("   FAIL: expected held-out AUROC > 0.70 (ceiling is 0.833)\n"); fails++; }
    auto ps = Parse("PREDICT EXISTS(events) FOR users WHERE id < 5 USING MODEL m");
    auto preds = RunPredict(db, m, ps);
    printf("   predicted %zu rows; p[0]=%.4f\n", preds.size(), preds.empty()?-1:preds[0].value);
    if (preds.size() != 5) { printf("   FAIL: expected 5 filtered rows\n"); fails++; }
    for (auto &p : preds) if (p.value < 0 || p.value > 1) { printf("   FAIL: prob out of range\n"); fails++; break; }
  }
  {
    printf("== B. leak canary (future independent of past)\n");
    // One small sample cannot separate a leak from noise, so average several
    // seeds at a size where the standard error is ~0.02.
    double acc = 0; int reps = 5;
    for (int k = 0; k < reps; k++) {
      Database db = MakeDb(false, 11 + (uint32_t)k * 7, 3000);
      auto st = Parse("TRAIN MODEL m PREDICT EXISTS(events) FOR users AT ts HORIZON 30 DAYS "
                      "OPTIONS (epochs = 60, hidden = 32, lr = 0.02)");
      Model m; auto rep = TrainModel(db, st, m);
      printf("   seed %d: test=%.4f (n=%zu)\n", k, rep.test_metric, rep.n_test);
      acc += rep.test_metric;
    }
    double mean = acc / reps;
    printf("   mean test AUROC = %.4f (must sit near 0.5)\n", mean);
    if (mean > 0.55) { printf("   FAIL: signal without a cause => future leaked into features\n"); fails++; }
  }
  {
    printf("== C. COUNT target is regression\n");
    Database db = MakeDb(true, 3);
    auto st = Parse("TRAIN MODEL c PREDICT COUNT(events) FOR users AT ts HORIZON 60 DAYS "
                    "OPTIONS (epochs = 60, hidden = 32)");
    Model m; auto rep = TrainModel(db, st, m);
    printf("   metric=%s val=%.4f classification=%d\n", rep.metric_name.c_str(), rep.val_metric, (int)m.classification);
  }
  {
    printf("== D. inner filter narrows the label\n");
    Database db = MakeDb(true, 5);
    auto st = Parse("TRAIN MODEL f PREDICT EXISTS(events WHERE amount > 12.5) FOR users AT ts "
                    "HORIZON 30 DAYS OPTIONS (epochs = 60, hidden = 32)");
    Model m; auto rep = TrainModel(db, st, m);
    printf("   trained on %zu rows, val=%.4f\n", rep.n_train, rep.val_metric);
    if (rep.n_train == 0) { printf("   FAIL: filter removed everything\n"); fails++; }
  }
  {
    printf("== E. GraphSAGE: learns real signal\n");
    Database db = MakeDb(true, 7);
    auto st = Parse("TRAIN MODEL sg PREDICT EXISTS(events) FOR users AT ts HORIZON 30 DAYS "
                    "OPTIONS (epochs = 120, hidden = 32, lr = 0.02, arch = 'sage')");
    Model m; auto rep = TrainModel(db, st, m);
    printf("   test=%.4f params=%d (ceiling 0.833)\n", rep.test_metric, rep.width);
    if (rep.test_metric < 0.68) { printf("   FAIL: sage did not learn\n"); fails++; }
  }
  {
    printf("== F. GraphSAGE leak canary\n");
    double acc = 0; int reps = 4;
    for (int k = 0; k < reps; k++) {
      Database db = MakeDb(false, 31 + (uint32_t)k * 5, 2500);
      auto st = Parse("TRAIN MODEL sg PREDICT EXISTS(events) FOR users AT ts HORIZON 30 DAYS "
                      "OPTIONS (epochs = 50, hidden = 32, lr = 0.02, arch = 'sage')");
      Model m; auto rep = TrainModel(db, st, m);
      acc += rep.test_metric;
    }
    double mean = acc / reps;
    printf("   mean test AUROC = %.4f (must sit near 0.5)\n", mean);
    if (mean > 0.57) { printf("   FAIL: sage leaks the future\n"); fails++; }
  }
  {
    // users -> orders -> items. Order rows carry NO signal of their own; the
    // label depends on how many ITEMS a user's orders contain. One hop can only
    // see order columns, so it cannot represent this; two hops can.
    printf("== G. two-hop signal (one hop provably cannot reach it)\n");
    std::mt19937 rng(5); std::uniform_real_distribution<double> U(0,1);
    const size_t NU = 700;
    Database db;
    Frame users; users.name = "users";
    Column uid = MakeCol("id", ColType::INT64, NU), uts = MakeCol("ts", ColType::TIMESTAMP, NU),
           uf = MakeCol("f", ColType::DOUBLE, NU);
    std::vector<double> anch(NU); std::vector<int> heavy(NU);
    for (size_t i = 0; i < NU; i++) {
      uid.num[i] = double(i); anch[i] = (900.0 + U(rng)*80) * DAY; uts.num[i] = anch[i];
      uf.num[i] = U(rng); heavy[i] = (U(rng) < 0.5) ? 1 : 0;
    }
    users.columns = {uid, uts, uf}; users.nrows = NU;
    // every user has exactly 3 past orders: order count carries no signal
    std::vector<double> o_id, o_uid, o_ts, o_c; std::vector<double> it_id, it_oid, it_ts, it_v;
    size_t onext = 0, inext = 0;
    for (size_t i = 0; i < NU; i++) {
      for (int k = 0; k < 3; k++) {
        const double ots = anch[i] - (double(k) + 1.0) * 10.0 * DAY;
        o_id.push_back(double(onext)); o_uid.push_back(double(i)); o_ts.push_back(ots); o_c.push_back(1.0);
        const int nitems = heavy[i] ? 9 : 1;   // the signal lives here, two hops out
        for (int j = 0; j < nitems; j++) {
          it_id.push_back(double(inext++)); it_oid.push_back(double(onext));
          it_ts.push_back(ots - 1.0*DAY); it_v.push_back(1.0);
        }
        onext++;
      }
      if (U(rng) < (heavy[i] ? 0.85 : 0.15)) {   // future order, driven by heaviness
        o_id.push_back(double(onext++)); o_uid.push_back(double(i));
        o_ts.push_back(anch[i] + 5.0*DAY); o_c.push_back(1.0);
      }
    }
    Frame orders; orders.name = "orders";
    Column oi = MakeCol("id", ColType::INT64, o_id.size()), ou = MakeCol("user_id", ColType::INT64, o_id.size()),
           ot = MakeCol("ts", ColType::TIMESTAMP, o_id.size()), oc = MakeCol("c", ColType::DOUBLE, o_id.size());
    for (size_t k = 0; k < o_id.size(); k++) { oi.num[k]=o_id[k]; ou.num[k]=o_uid[k]; ot.num[k]=o_ts[k]; oc.num[k]=o_c[k]; }
    orders.columns = {oi, ou, ot, oc}; orders.nrows = o_id.size();
    Frame items; items.name = "items";
    Column ii = MakeCol("id", ColType::INT64, it_id.size()), io = MakeCol("order_id", ColType::INT64, it_id.size()),
           it = MakeCol("ts", ColType::TIMESTAMP, it_id.size()), iv = MakeCol("v", ColType::DOUBLE, it_id.size());
    for (size_t k = 0; k < it_id.size(); k++) { ii.num[k]=it_id[k]; io.num[k]=it_oid[k]; it.num[k]=it_ts[k]; iv.num[k]=it_v[k]; }
    items.columns = {ii, io, it, iv}; items.nrows = it_id.size();
    db.tables = {users, orders, items};
    db.fks.push_back({"orders", "user_id", "users", "id"});
    db.fks.push_back({"items", "order_id", "orders", "id"});
    db.BuildLinks({});
    const char *opts1 = "OPTIONS (epochs = 100, hidden = 32, lr = 0.02, arch = 'sage', layers = 1)";
    const char *opts2 = "OPTIONS (epochs = 100, hidden = 32, lr = 0.02, arch = 'sage', layers = 2)";
    double a1 = 0, a2 = 0;
    { auto st = Parse(std::string("TRAIN MODEL h1 PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts1);
      Model m; a1 = TrainModel(db, st, m).test_metric; }
    { auto st = Parse(std::string("TRAIN MODEL h2 PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts2);
      Model m; a2 = TrainModel(db, st, m).test_metric; }
    printf("   1 hop = %.4f   2 hops = %.4f\n", a1, a2);
    if (a2 <= a1 + 0.03) { printf("   FAIL: the second hop did not reach the signal\n"); fails++; }

    // ---- H. the same statement twice in a session gives the same model -----
    // The SAGE optimiser state used to be static and was only reset when it had
    // to grow, so a second TRAIN inherited the first model's Adam moments and
    // step count. Nothing about the statement changed, but the result did.
    printf("== H. training is reproducible within a session\n");
    double r1 = 0, r2 = 0, r3 = 0;
    { auto st = Parse(std::string("TRAIN MODEL r PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts2);
      Model m; r1 = TrainModel(db, st, m).test_metric; }
    { auto st = Parse(std::string("TRAIN MODEL other PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts1);
      Model m; TrainModel(db, st, m); }          // a different model in between
    { auto st = Parse(std::string("TRAIN MODEL r PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts2);
      Model m; r2 = TrainModel(db, st, m).test_metric; }
    { auto st = Parse(std::string("TRAIN MODEL r PREDICT EXISTS(orders) FOR users AT ts HORIZON 20 DAYS ") + opts2);
      Model m; r3 = TrainModel(db, st, m).test_metric; }
    printf("   %.6f / %.6f / %.6f\n", r1, r2, r3);
    if (r1 != r2 || r2 != r3) {
      printf("   FAIL: the same statement gave different models\n"); fails++;
    }
  }
  printf("\n%s\n", fails ? "FAILURES" : "all model tests passed");
  return fails ? 1 : 0;
}
