#ifndef __CPU_TIMER_H__
#define __CPU_TIMER_H__

#include <windows.h>

struct CpuTimer
{
    LARGE_INTEGER tic, toc, freq;

    CpuTimer()
    {
        QueryPerformanceFrequency(&freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&tic);
    }

    void Stop()
    {
        QueryPerformanceCounter(&toc);
    }

    double Elapsed()
    {
        return (toc.QuadPart - tic.QuadPart) / (double)freq.QuadPart;
    }
};

#endif /* __CPU_TIMER_H__ */