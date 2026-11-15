// Read bandwidth of the Pi's memory, the ceiling for writing tokens (each token reads every weight once).
// Sums a large buffer (far larger than the caches) with 1..N threads, best of several passes.
//   gcc -O3 -march=native -pthread pi/membw.c -o membw && ./membw 2048 4      (MiB, max threads)
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t *buf;
static size_t n;
static int threads;

typedef struct { int id; uint64_t sum; } arg_t;

static void *reader(void *p) {
  arg_t *a = p;
  size_t lo = n * a->id / threads, hi = n * (a->id + 1) / threads;
  uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (size_t i = lo; i + 4 <= hi; i += 4) { s0 += buf[i]; s1 += buf[i + 1]; s2 += buf[i + 2]; s3 += buf[i + 3]; }
  a->sum = s0 + s1 + s2 + s3;
  return NULL;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
  size_t mib = argc > 1 ? strtoul(argv[1], 0, 10) : 2048;
  int maxt = argc > 2 ? atoi(argv[2]) : 4;
  n = mib * 1024 * 1024 / sizeof(uint64_t);
  buf = aligned_alloc(64, n * sizeof(uint64_t));
  for (size_t i = 0; i < n; i++) buf[i] = i;  // touch every page
  uint64_t keep = 0;
  for (threads = 1; threads <= maxt; threads++) {
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
      pthread_t t[64]; arg_t a[64];
      double t0 = now();
      for (int i = 0; i < threads; i++) { a[i].id = i; pthread_create(&t[i], 0, reader, &a[i]); }
      for (int i = 0; i < threads; i++) { pthread_join(t[i], 0); keep += a[i].sum; }
      double gbs = n * sizeof(uint64_t) / (now() - t0) / 1e9;
      if (gbs > best) best = gbs;
    }
    printf("threads %d: read %.2f GB/s\n", threads, best);
  }
  return keep == 42;  // keeps the sums from being optimised away
}
