// Parser fuzzer. Mutates valid statements and feeds garbage, checking three
// things that are easy to get wrong and hard to notice:
//
//   1. Parse either returns or throws. It never crashes and never hangs.
//   2. Anything it accepts survives ToString -> Parse unchanged. A statement is
//      stored as a model's record of itself and echoed by SHOW MODELS, so one
//      that cannot be read back is a statement that has quietly changed meaning.
//   3. A rejection carries a position inside the input.
//
// Run it under sanitizers:
//   make -C extras/pql sanitize   (or: make -C extras/pql test_fuzz)
//
// This is how the unterminated-quote bug was found: `region = 'US` ran to the
// end of the input and was accepted as `region = 'US'`.
#include "pql.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
using namespace pql;

static const char *kSeeds[] = {
  "TRAIN MODEL m PREDICT COUNT(orders) FOR users AT ts HORIZON 30 DAYS",
  "TRAIN OR REPLACE MODEL m PREDICT EXISTS(o WHERE o.k = 'x') FOR u AS uu WHERE uu.r = 'EU' "
  "EXCLUDE (email, o.note) AT ts HORIZON 7 DAYS USING GRAPH (u, o) "
  "SPLIT TEMPORAL VALIDATE FROM 100 TEST FROM 200 OPTIONS (epochs = 7, arch = 'sage')",
  "PREDICT users.churn FOR users WHERE id IN (1, 2, 3) USING MODEL m",
  "PREDICT SUM(orders.amount) FOR customers AT '2024-01-01' USING MODEL spend",
  "BACKTEST MODEL m FOR users WHERE x > 1 FROM '2024-01-01' TO '2024-06-01'",
  "EXPLAIN MODEL m FOR users WHERE a IS NOT NULL",
  "TRAIN MODEL g PREDICT COUNT(o) FOR p EVERY 7 DAYS HORIZON 30 DAYS",
  "SHOW MODELS", "DROP MODEL m",
  "PREDICT users.x FOR users WHERE NOT (a = 1 OR b < 2) AND c >= 3 USING MODEL m",
  nullptr };

static std::vector<std::string> Tokens(const std::string &s) {
  std::vector<std::string> out; std::string cur;
  for (char c : s) {
    if (c == ' ') { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
    else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

int main(int argc, char **argv) {
  // test_fuzz [iterations] [seed]: the defaults are the quick check; a soak
  // run under sanitizers loops over seeds.
  const long iters = argc > 1 ? std::atol(argv[1]) : 60000;
  std::mt19937 rng(argc > 2 ? (unsigned)std::atol(argv[2]) : 12345u);
  // Every keyword in the grammar, every literal shape, and the things a
  // tokenizer gets wrong: unterminated quotes, unicode, very long identifiers,
  // numbers at the edge of int64 and doubles, deep nesting.
  static const std::string long_ident(5000, 'z');
  static const std::string deep_open(300, '(');
  static const std::string deep_close(300, ')');
  static const std::string many_nots = [] { std::string s; for (int i = 0; i < 300; i++) s += "NOT "; return s; }();
  const char *alphabet[] = {"(", ")", ",", "'", "\"", "=", ";", "*", "-", ".", "0", "99",
                            "TRAIN", "MODEL", "PREDICT", "FOR", "AT", "HORIZON", "DAYS",
                            "WHERE", "OPTIONS", "EXCLUDE", "SPLIT", "x", "'s", "1e999",
                            "--", "/*", "\\", "\t", "\n",
                            "OR", "REPLACE", "AND", "NOT", "IN", "IS", "NULL", "AS", "IF",
                            "EXISTS", "COUNT", "SUM", "AVG", "MIN", "MAX", "EVERY", "USING",
                            "GRAPH", "TEMPORAL", "VALIDATE", "TEST", "FROM", "TO", "BACKTEST",
                            "EXPLAIN", "SHOW", "MODELS", "DROP", "WEEKS", "MONTHS", "YEARS",
                            "HOURS", "DAY", "<", ">", "<=", ">=", "!=", "<>", "==", "+", "/",
                            "1.2.3", "99999999999999999999", ".5", "1e5", "-0", "0.0.0",
                            "1000000000000", "'", "''", "'a''b'", "\"q\"", "\"\"",
                            "\xc3\xa9", "\xe2\x82\xac", "\xff", "\x00",
                            long_ident.c_str(), deep_open.c_str(), deep_close.c_str(),
                            many_nots.c_str(), nullptr};
  int nalpha = 0; while (alphabet[nalpha]) nalpha++;
  long parsed = 0, rejected = 0, roundtrip_fail = 0, badpos = 0, other = 0;
  for (long iter = 0; iter < iters; iter++) {
    // build an input: a mutated seed, or pure noise
    std::string in;
    if (iter % 4 == 3) {
      const int n = 1 + (int)(rng() % 12);
      for (int i = 0; i < n; i++) { in += alphabet[rng() % nalpha]; in += " "; }
    } else {
      auto tk = Tokens(kSeeds[rng() % 10]);
      const int muts = 1 + (int)(rng() % 4);
      for (int m = 0; m < muts && !tk.empty(); m++) {
        const size_t at = rng() % tk.size();
        switch (rng() % 4) {
        case 0: tk.erase(tk.begin() + (long)at); break;                    // delete
        case 1: tk.insert(tk.begin() + (long)at, alphabet[rng() % nalpha]); break;
        case 2: tk[at] = alphabet[rng() % nalpha]; break;                   // replace
        case 3: tk.resize(at); break;                                       // truncate
        }
      }
      for (size_t i = 0; i < tk.size(); i++) { in += tk[i]; in += " "; }
    }
    try {
      Statement st = Parse(in);
      parsed++;
      const std::string once = st.ToString();
      try {
        Statement again = Parse(once);
        if (again.ToString() != once) {
          if (roundtrip_fail < 5) {
            printf("ROUND-TRIP DRIFT\n  in:    %s\n  once:  %s\n  twice: %s\n",
                   in.c_str(), once.c_str(), again.ToString().c_str());
          }
          roundtrip_fail++;
        }
      } catch (const std::exception &e) {
        if (roundtrip_fail < 5) {
          printf("ACCEPTED BUT ITS OWN ToString IS REJECTED\n  in:   %s\n  out:  %s\n  err:  %s\n",
                 in.c_str(), once.c_str(), e.what());
        }
        roundtrip_fail++;
      }
    } catch (const ParseError &e) {
      rejected++;
      if (e.position > in.size() + 1) { badpos++; }
    } catch (const std::exception &) {
      other++;   // semantic refusals from Validate, also fine
    }
  }
  printf("parsed %ld, rejected %ld, other refusals %ld\n", parsed, rejected, other);
  printf("round-trip failures %ld, error positions outside the input %ld\n",
         roundtrip_fail, badpos);
  return (roundtrip_fail || badpos) ? 1 : 0;
}
