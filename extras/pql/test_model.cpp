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
  // ---- I. a text column is a feature, and an unseen value is handled --------
  // The label is decided entirely by a category. Nothing numeric predicts it, so
  // a model that cannot read text can only return the mean.
  {
    printf("== I. categorical features\n");
    const size_t NU = 3000;
    Database db;
    Frame users; users.name = "users";
    Column uid = MakeCol("id", ColType::INT64, NU), uts = MakeCol("ts", ColType::TIMESTAMP, NU),
           filler = MakeCol("filler", ColType::DOUBLE, NU),
           plan = MakeCol("plan", ColType::CATEGORY, NU);
    const char *names[3] = {"premium", "basic", "trial"};
    const int per_plan[3] = {18, 4, 0};
    std::vector<double> anch(NU); std::vector<int> pidx(NU);
    for (size_t i = 0; i < NU; i++) {
      uid.num[i] = double(i);
      anch[i] = (900.0 + double(i % 400)) * DAY;
      uts.num[i] = anch[i];
      filler.num[i] = 0.5;                       // constant: carries nothing
      pidx[i] = int(i % 3);
      plan.code[i] = plan.dict.Intern(names[pidx[i]]);
    }
    users.columns = {uid, uts, filler, plan}; users.nrows = NU;
    std::vector<double> o_id, o_uid, o_ts;
    size_t onext = 0;
    for (size_t i = 0; i < NU; i++) {
      for (int k = 0; k < per_plan[pidx[i]]; k++) {
        o_id.push_back(double(onext++)); o_uid.push_back(double(i));
        o_ts.push_back(anch[i] + (1.0 + double(k % 25)) * DAY);
      }
    }
    Frame orders; orders.name = "orders";
    Column oi = MakeCol("id", ColType::INT64, o_id.size()),
           ou = MakeCol("user_id", ColType::INT64, o_id.size()),
           ot = MakeCol("ts", ColType::TIMESTAMP, o_id.size());
    for (size_t k = 0; k < o_id.size(); k++) { oi.num[k]=o_id[k]; ou.num[k]=o_uid[k]; ot.num[k]=o_ts[k]; }
    orders.columns = {oi, ou, ot}; orders.nrows = o_id.size();
    db.tables = {users, orders};
    db.fks.push_back({"orders", "user_id", "users", "id"});
    db.BuildLinks({});
    auto st = Parse("TRAIN MODEL cat PREDICT COUNT(orders) FOR users AT ts HORIZON 30 DAYS "
                    "OPTIONS (epochs = 60, hidden = 32)");
    Model m; TrainReport rep = TrainModel(db, st, m);
    printf("   mae = %.4f   baseline = %.4f   features = %d\n",
           rep.test_metric, rep.baseline_metric, rep.width);
    if (rep.test_metric > 0.5) {
      printf("   FAIL: the category was not learned (a mean-only model scores ~%.2f)\n",
             rep.baseline_metric);
      fails++;
    }

    // Both GraphSAGE depths read the same entity columns. Two hops used to skip
    // them entirely: its gather never filled the self block, so w_self took a
    // zero gradient every step and the entity's own attributes were invisible.
    for (int layers = 1; layers <= 2; layers++) {
      char q[240];
      snprintf(q, sizeof q,
               "TRAIN MODEL sg PREDICT COUNT(orders) FOR users AT ts HORIZON 30 DAYS "
               "OPTIONS (epochs = 60, hidden = 32, arch = 'sage', layers = %d)", layers);
      auto sst = Parse(q);
      Model sm; TrainReport srep = TrainModel(db, sst, sm);
      printf("   sage layers=%d mae = %.4f (baseline %.4f)\n", layers, srep.test_metric,
             srep.baseline_metric);
      if (srep.test_metric > 0.5) {
        printf("   FAIL: sage layers=%d did not read the entity's text column\n", layers);
        fails++;
      }
    }

    // EXPLAIN must name the column that actually drives the label, and must put
    // the constant one at the bottom. This is the check that would have caught a
    // feature being wired up but never reaching the model.
    {
      auto ex = Parse("EXPLAIN MODEL cat");
      auto imp = RunExplain(db, m, ex);
      if (imp.empty()) {
        printf("   FAIL: EXPLAIN returned nothing\n"); fails++;
      } else {
        double filler_drop = 0; bool found = false;
        for (const auto &f : imp) {
          if (f.feature.rfind("filler", 0) == 0) { filler_drop = f.drop; found = true; }
        }
        printf("   explain: top is '%s' at %.3f; filler at %.4f\n",
               imp[0].feature.c_str(), imp[0].drop, filler_drop);
        if (imp[0].feature.rfind("plan", 0) != 0) {
          printf("   FAIL: the driving column is not ranked first\n"); fails++;
        }
        if (!found || std::fabs(filler_drop) > 0.05) {
          printf("   FAIL: a constant column should cost nothing to shuffle\n"); fails++;
        }
      }
      // A SAGE model has no dense feature vector to permute, and says so.
      auto sst = Parse("TRAIN MODEL sg2 PREDICT COUNT(orders) FOR users AT ts HORIZON 30 DAYS "
                       "OPTIONS (epochs = 5, hidden = 16, arch = 'sage')");
      Model sm; TrainModel(db, sst, sm);
      bool refused = false;
      try {
        auto e2 = Parse("EXPLAIN MODEL sg2");
        RunExplain(db, sm, e2);
      } catch (const std::exception &) {
        refused = true;
      }
      if (!refused) {
        printf("   FAIL: explaining a sage model should refuse, not invent numbers\n"); fails++;
      }
    }

    // A value never seen in training must land in the spare slot, not crash and
    // not be mistaken for one of the trained categories.
    Frame &u2 = db.tables[0];
    Column &p2 = u2.columns[3];
    const uint32_t fresh = p2.dict.Intern("enterprise");
    for (size_t i = 0; i < 40; i++) { p2.code[i] = fresh; }
    auto ps = Parse("PREDICT COUNT(orders) FOR users USING MODEL cat");
    std::vector<Prediction> preds = RunPredict(db, m, ps);
    double lo = 1e18, hi = -1e18;
    for (size_t i = 0; i < preds.size() && i < 40; i++) {
      lo = std::min(lo, preds[i].value); hi = std::max(hi, preds[i].value);
    }
    printf("   unseen category predicts %.3f..%.3f over %zu rows\n", lo, hi, preds.size());
    if (!(lo == lo) || !(hi == hi) || preds.size() != NU) {
      printf("   FAIL: an unseen category did not survive prediction\n"); fails++;
    }
  }

  // ---- J. two things the aggregates alone cannot express ------------------
  // Each task splits users into groups that differ in exactly one thing. If the
  // features reach it, the mean prediction per group separates; if not, every
  // group gets the same number, which an MAE alone would not make obvious.
  {
    printf("== J. seasonality and regularity\n");
    struct Case { const char *name; int kind; };
    const Case cases[2] = {{"anchor day-of-week", 0}, {"gap regularity", 1}};
    for (const Case &cs : cases) {
      std::mt19937 rng(3); std::uniform_real_distribution<double> U(0, 1);
      const size_t NU = 3000;
      Database db;
      Frame users; users.name = "users";
      Column uid = MakeCol("id", ColType::INT64, NU), uts = MakeCol("ts", ColType::TIMESTAMP, NU),
             fil = MakeCol("filler", ColType::DOUBLE, NU);
      std::vector<double> anch(NU); std::vector<int> grp(NU);
      std::vector<double> e_uid, e_ts;
      for (size_t i = 0; i < NU; i++) {
        int nf = 0;
        if (cs.kind == 0) {
          const int day = (int)(U(rng) * 280.0);
          anch[i] = (1000.0 + double(day)) * DAY;
          for (int k = 0; k < 4; k++) {                     // identical history
            e_uid.push_back(double(i)); e_ts.push_back(anch[i] - (5.0 + double(k) * 7.0) * DAY);
          }
          const int dow = day % 7;
          grp[i] = (dow == 0 || dow == 6) ? 0 : 1;
          nf = grp[i] ? 9 : 1;
        } else {
          anch[i] = (1000.0 + U(rng) * 300.0) * DAY;
          grp[i] = (int)(U(rng) * 2);
          // Eight events, all inside (7, 30] days back, so every window count and
          // every recency matches. Only the spacing differs.
          const double regular[8] = {8, 11, 14, 17, 20, 23, 26, 29};
          const double bursty[8] = {8, 9, 10, 11, 26, 27, 28, 29};
          for (int k = 0; k < 8; k++) {
            e_uid.push_back(double(i));
            e_ts.push_back(anch[i] - (grp[i] ? regular[k] : bursty[k]) * DAY);
          }
          nf = grp[i] ? 8 : 1;
        }
        for (int k = 0; k < nf; k++) {
          e_uid.push_back(double(i)); e_ts.push_back(anch[i] + (1.0 + U(rng) * 6.0) * DAY);
        }
        uid.num[i] = double(i); uts.num[i] = anch[i]; fil.num[i] = 0.5;
      }
      users.columns = {uid, uts, fil}; users.nrows = NU;
      Frame ev; ev.name = "events";
      const size_t NE = e_uid.size();
      Column vi = MakeCol("id", ColType::INT64, NE), vu = MakeCol("user_id", ColType::INT64, NE),
             vt = MakeCol("ts", ColType::TIMESTAMP, NE);
      for (size_t k = 0; k < NE; k++) { vi.num[k]=double(k); vu.num[k]=e_uid[k]; vt.num[k]=e_ts[k]; }
      ev.columns = {vi, vu, vt}; ev.nrows = NE;
      db.tables = {users, ev};
      db.fks.push_back({"events", "user_id", "users", "id"});
      db.BuildLinks({});
      const char *hz = cs.kind == 0 ? "7" : "30";
      char q[260];
      snprintf(q, sizeof q,
               "TRAIN MODEL j PREDICT COUNT(events) FOR users AT ts HORIZON %s DAYS "
               "OPTIONS (epochs = 80, hidden = 64)", hz);
      auto st = Parse(q);
      Model m; TrainModel(db, st, m);
      auto ps = Parse("PREDICT COUNT(events) FOR users USING MODEL j");
      std::vector<Prediction> preds = RunPredict(db, m, ps);
      double sum[2] = {0, 0}; int cnt[2] = {0, 0};
      for (const auto &p : preds) { sum[grp[p.entity_row]] += p.value; cnt[grp[p.entity_row]]++; }
      const double m0 = cnt[0] ? sum[0] / cnt[0] : 0, m1 = cnt[1] ? sum[1] / cnt[1] : 0;
      printf("   %-20s quiet group %.2f, busy group %.2f\n", cs.name, m0, m1);
      if (m1 - m0 < 3.0) {
        printf("   FAIL: the two groups got the same answer, so the signal is unreachable\n");
        fails++;
      }
    }
  }

  // ---- K. average precision against hand-computed values ------------------
  {
    printf("== K. average precision\n");
    struct Case { std::vector<std::pair<double,int>> in; double want; const char *what; };
    const std::vector<Case> cases = {
      {{{0.9,1},{0.8,1},{0.7,0},{0.6,0}}, 1.0,          "perfect ranking"},
      {{{0.9,1},{0.8,0},{0.7,1},{0.6,0}}, 5.0/6.0,      "one negative interleaved"},
      {{{0.9,0},{0.8,0},{0.7,1},{0.6,1}}, (1.0/3+0.5)/2,"reversed ranking"},
      {{{0.5,1},{0.5,0},{0.5,1},{0.5,0}}, 0.5,          "all tied"},
    };
    for (const auto &c : cases) {
      const double got = AveragePrecision(c.in);
      printf("   %-24s got %.4f want %.4f\n", c.what, got, c.want);
      if (std::fabs(got - c.want) > 1e-9) {
        printf("   FAIL\n"); fails++;
      }
    }
    // Undefined when one class is missing, rather than silently zero.
    const double none = AveragePrecision({{0.9,0},{0.8,0}});
    const double all = AveragePrecision({{0.9,1},{0.8,1}});
    if (!(none != none) || !(all != all)) {
      printf("   FAIL: a single-class fold must be undefined, got %.4f / %.4f\n", none, all);
      fails++;
    }
  }

  printf("\n%s\n", fails ? "FAILURES" : "all model tests passed");
  return fails ? 1 : 0;
}
