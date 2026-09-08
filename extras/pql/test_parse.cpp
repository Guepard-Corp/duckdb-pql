#include "src/pql.hpp"
#include <cstdio>
static int fails = 0, oks = 0;
static void Ok(const char *sql, const char *expect_substr = nullptr) {
  try {
    auto st = pql::Parse(sql);
    std::string s = st.ToString();
    if (expect_substr && s.find(expect_substr) == std::string::npos) {
      std::printf("  FAIL roundtrip: %s\n     got: %s\n     want substr: %s\n", sql, s.c_str(), expect_substr);
      fails++; return;
    }
    oks++;
  } catch (const std::exception &e) {
    std::printf("  FAIL parse: %s\n     error: %s\n", sql, e.what()); fails++;
  }
}
static void Err(const char *sql, const char *want) {
  try { pql::Parse(sql); std::printf("  FAIL (expected error): %s\n", sql); fails++; }
  catch (const std::exception &e) {
    std::string m = e.what();
    if (m.find(want) == std::string::npos) { std::printf("  FAIL wrong error: %s\n     got: %s\n     want: %s\n", sql, m.c_str(), want); fails++; }
    else oks++;
  }
}
int main() {
  std::printf("== accepts\n");
  Ok("PREDICT users.churn FOR users USING MODEL m", "users.churn");
  Ok("PREDICT users.churn FOR users AS u WHERE u.id = 42 USING MODEL m", "u.id = 42");
  Ok("PREDICT COUNT(orders) FOR customers AS c AT c.last_seen HORIZON 30 DAYS USING MODEL m", "COUNT(orders)");
  Ok("PREDICT EXISTS(votes) FOR users AT ts HORIZON 3 MONTHS USING MODEL m", "3 MONTHS");
  Ok("PREDICT SUM(orders.amount) FOR users AT ts HORIZON 1 YEAR USING MODEL m", "SUM(orders.amount)");
  Ok("TRAIN MODEL churn PREDICT EXISTS(orders) FOR customers AS c AT c.seen HORIZON 90 DAYS "
     "WHERE c.region = 'EU' SPLIT TEMPORAL VALIDATE FROM '2023-01-01' TEST FROM '2023-07-01' "
     "OPTIONS (epochs = 40, hidden = 128)", "TRAIN MODEL churn");
  Ok("PREDICT COUNT(events WHERE events.status IN ('yes','maybe')) FOR users AT t HORIZON 7 DAYS USING MODEL m", "IN ('yes', 'maybe')");
  Ok("PREDICT users.x FOR users WHERE a = 1 AND (b > 2 OR c IS NOT NULL) USING MODEL m", "OR");
  Ok("PREDICT users.x FOR users u WHERE u.k != 'z' USING MODEL m", "u.k != 'z'");  // implicit alias
  Ok("PREDICT COUNT(*) FOR users AT t HORIZON 1 WEEK USING MODEL m", "COUNT(*)");
  Ok("TRAIN MODEL m PREDICT COUNT(o) FOR u AT t HORIZON 5 DAYS USING GRAPH (a, b, c)", "USING GRAPH (a, b, c)");
  Ok("SHOW MODELS", "SHOW MODELS");
  Ok("DROP MODEL churn", "DROP MODEL churn");
  Ok("-- comment\nPREDICT users.churn FOR users USING MODEL m", "users.churn");
  Ok("PREDICT \"users\".\"select\" FOR users USING MODEL m", "select");  // quoted keyword as ident
  std::printf("== rejects\n");
  Err("TRAIN MODEL m PREDICT COUNT(orders) FOR users AT t", "HORIZON");
  Err("TRAIN MODEL m PREDICT COUNT(orders) FOR users HORIZON 5 DAYS", "anchor");
  Err("PREDICT users.x FOR users HORIZON 5 DAYS USING MODEL m", "meaningless");
  Err("PREDICT users.x FOR users", "USING MODEL");
  Err("TRAIN MODEL m PREDICT COUNT(o) FOR u AT '2024-01-01' HORIZON 5 DAYS", "per-row anchor");
  Err("PREDICT SUM(orders) FOR u AT t HORIZON 5 DAYS USING MODEL m", "needs a column");
  Err("PREDICT users.x FOR users WHERE a = b USING MODEL m", "single quotes");
  Err("PREDICT users.x FOR users WHERE a = 1 WHERE b = 2 USING MODEL m", "duplicate WHERE");
  Err("PREDICT COUNT(o) FOR u AT t HORIZON 5 PARSECS USING MODEL m", "unknown horizon unit");
  Err("TRAIN MODEL a PREDICT COUNT(o) FOR u AT t HORIZON 5 DAYS USING MODEL b", "USING MODEL belongs");
  Err("TRAIN MODEL a PREDICT COUNT(o) FOR u AT t HORIZON 5 DAYS OPTIONS (epocs = 10)", "unknown option");
  Err("TRAIN MODEL a PREDICT COUNT(o) FOR u AT t HORIZON 0 DAYS", "positive");
  Err("PREDICT users.x FOR users OPTIONS (epochs = 5) USING MODEL m", "OPTIONS apply to TRAIN");
  Err("PREDICT users.x FOR users SPLIT TEMPORAL VALIDATE FROM 1 TEST FROM 2 USING MODEL m", "SPLIT applies to TRAIN");
  Err("PREDICT COUNT(o) FOR u USING GRAPH (a) USING MODEL m", "USING GRAPH applies to TRAIN");
  Ok("TRAIN MODEL a PREDICT COUNT(o) FOR u AT t HORIZON 5 DAYS OPTIONS (mean_cols = 4, batch = 32)", "TRAIN MODEL a");
  std::printf("\n%d passed, %d failed\n", oks, fails);
  return fails ? 1 : 0;
}
