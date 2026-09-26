/* SIGSEGV/SIGBUS/SIGABRT/SIGILL handler that prints a native backtrace with library+offset for
 * every frame, then re-raises with the default action. There is no gdb on this cluster, and a crash
 * on a thread with no Python frame is invisible to faulthandler. Loaded via ctypes
 * (../adaptive_train_memtrace.py, SEGV_BT=1); name the frames with symbolize.py. Written for
 * experiments_md/20260926_05. Build inside the container (the .so is gitignored):
 *
 *   gcc -O0 -g -fPIC -shared -o segv_bt.so segv_bt.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/syscall.h>

static char altstack[1 << 16];

static void put(const char *s) { (void)!write(2, s, strlen(s)); }

static void handler(int sig, siginfo_t *si, void *uc_) {
    char buf[512];
    ucontext_t *uc = (ucontext_t *)uc_;
    void *pc = (void *)uc->uc_mcontext.gregs[REG_RIP];
    snprintf(buf, sizeof buf, "\n=== SEGV_BT signal %d addr %p pc %p tid %ld ===\n", sig, si->si_addr, pc,
             (long)syscall(SYS_gettid));
    put(buf);
    void *frames[64];
    int n = backtrace(frames, 64);
    /* frames[0..1] are this handler and the signal trampoline; also print the faulting pc. */
    void *all[65];
    all[0] = pc;
    for (int i = 0; i < n; i++) all[i + 1] = frames[i];
    for (int i = 0; i < n + 1; i++) {
        Dl_info info;
        if (dladdr(all[i], &info) && info.dli_fname) {
            snprintf(buf, sizeof buf, "#%02d %p %s+0x%lx (%s+0x%lx)\n", i, all[i], info.dli_fname,
                     (unsigned long)((char *)all[i] - (char *)info.dli_fbase),
                     info.dli_sname ? info.dli_sname : "?",
                     info.dli_saddr ? (unsigned long)((char *)all[i] - (char *)info.dli_saddr) : 0UL);
        } else {
            snprintf(buf, sizeof buf, "#%02d %p ?\n", i, all[i]);
        }
        put(buf);
    }
    put("=== SEGV_BT end ===\n");
    signal(sig, SIG_DFL);
    raise(sig);
}

int install(void) {
    stack_t ss = {.ss_sp = altstack, .ss_size = sizeof altstack, .ss_flags = 0};
    sigaltstack(&ss, NULL);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    return 0;
}
