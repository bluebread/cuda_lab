#include <windows.h>

double PCFreq = 0.0;
__int64 CounterStart = 0;

#define MEASURE_NO_RETURN(__func, ...)          \
	(StartCounter(),                            \
	__func(__VA_ARGS__)),                        \
	GetCounter()
     
void StartCounter()
{
    LARGE_INTEGER li;
    // if(!QueryPerformanceFrequency(&li)) return;

    PCFreq = double(li.QuadPart)/1000.0; // milliseconds

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}

#define MAX_THREADS_PER_BLOCK 1024
#define ITER_NUM 50

#ifndef TYPE
#define TYPE float
#endif

#ifndef OP 
#define OP +
#endif