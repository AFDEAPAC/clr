/* Copyright (c) 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <atomic>

namespace hip_debug {

struct LogState {
  FILE* file = nullptr;
  int enabled = -1;              // -1=not checked, 0=off, 1=on
  char path[512] = {};
  size_t max_bytes = 0;          // 0=unlimited
  std::atomic<size_t> written{0};
  int rotation_count = 0;
  pthread_mutex_t rotate_lock = PTHREAD_MUTEX_INITIALIZER;
};

inline LogState& state() {
  static LogState s;
  return s;
}

inline pthread_once_t& onceCtrl() {
  static pthread_once_t o = PTHREAD_ONCE_INIT;
  return o;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
inline void initLog() {
  auto& s = state();
  const char* env = getenv("HIP_DEBUG_LOG");
  if (!env || env[0] == '\0' || (env[0] == '0' && env[1] == '\0')) {
    s.enabled = 0;
    return;
  }

  if (strcmp(env, "1") == 0) {
    snprintf(s.path, sizeof(s.path), "/tmp/hip_debug_%d.log", getpid());
  } else {
    snprintf(s.path, sizeof(s.path), "%.*s", (int)(sizeof(s.path) - 1), env);
    char* pct = strstr(s.path, "%d");
    if (pct) {
      char tmp[512];
      *pct = '\0';
      snprintf(tmp, sizeof(tmp), "%s%d%s", s.path, getpid(), pct + 2);
      snprintf(s.path, sizeof(s.path), "%s", tmp);
    }
  }

  // HIP_DEBUG_LOG_MAX_MB: max log size in MB before rotation (default 10240 = 10GB, 0=unlimited)
  const char* max_env = getenv("HIP_DEBUG_LOG_MAX_MB");
  if (max_env && atoi(max_env) > 0) {
    s.max_bytes = (size_t)atoi(max_env) * 1024ULL * 1024ULL;
  } else {
    s.max_bytes = 10ULL * 1024 * 1024 * 1024;  // default 10GB
  }

  s.file = fopen(s.path, "a");
  if (s.file) {
    s.enabled = 1;
    // Get current file size
    struct stat st;
    if (fstat(fileno(s.file), &st) == 0) {
      s.written.store(st.st_size, std::memory_order_relaxed);
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(s.file, "[%ld.%06ld] === HIP Debug Log opened (pid=%d, max=%zuMB) ===\n",
            (long)ts.tv_sec, ts.tv_nsec / 1000, getpid(),
            s.max_bytes / (1024 * 1024));
    fflush(s.file);
  } else {
    s.enabled = 0;
  }
}
#pragma GCC diagnostic pop

// Rotate: close current log, compress with gzip in background, open new log
inline void rotateLog() {
  auto& s = state();
  pthread_mutex_lock(&s.rotate_lock);

  // Double-check after acquiring lock
  if (s.written.load(std::memory_order_relaxed) < s.max_bytes || s.max_bytes == 0) {
    pthread_mutex_unlock(&s.rotate_lock);
    return;
  }

  s.rotation_count++;
  fflush(s.file);
  fclose(s.file);
  s.file = nullptr;

  // Rename current log to .N and compress in background
  char rotated[560];
  snprintf(rotated, sizeof(rotated), "%s.%d", s.path, s.rotation_count);
  rename(s.path, rotated);

  // Fork a child to gzip compress (non-blocking)
  pid_t child = fork();
  if (child == 0) {
    // Child: compress the rotated log
    execlp("gzip", "gzip", "-f", rotated, nullptr);
    _exit(1);  // exec failed
  }
  // Parent: don't wait — gzip runs in background

  // Open new log file
  s.file = fopen(s.path, "a");
  s.written.store(0, std::memory_order_relaxed);

  if (s.file) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(s.file, "[%ld.%06ld] === Log rotated (rotation #%d, previous compressed to %s.gz) ===\n",
            (long)ts.tv_sec, ts.tv_nsec / 1000, s.rotation_count, rotated);
    fflush(s.file);
  } else {
    s.enabled = 0;
  }

  pthread_mutex_unlock(&s.rotate_lock);
}

inline void dlog(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
inline void dlog(const char* fmt, ...) {
  pthread_once(&onceCtrl(), initLog);
  auto& s = state();
  if (s.enabled <= 0 || !s.file) return;

  // Check if rotation needed (cheap relaxed check)
  if (s.max_bytes > 0 && s.written.load(std::memory_order_relaxed) >= s.max_bytes) {
    rotateLog();
    if (!s.file) return;
  }

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);

  char buf[1024];
  int prefix_len = snprintf(buf, sizeof(buf), "[%ld.%06ld] ",
                            (long)ts.tv_sec, ts.tv_nsec / 1000);

  va_list ap;
  va_start(ap, fmt);
  int msg_len = vsnprintf(buf + prefix_len, sizeof(buf) - prefix_len, fmt, ap);
  va_end(ap);

  if (msg_len < 0) return;
  int total = prefix_len + msg_len;
  if (total >= (int)sizeof(buf)) total = sizeof(buf) - 1;

  fwrite(buf, 1, total, s.file);
  fflush(s.file);
  s.written.fetch_add(total, std::memory_order_relaxed);
}

}  // namespace hip_debug

#define HIP_DLOG(fmt, ...) hip_debug::dlog(fmt, ##__VA_ARGS__)
