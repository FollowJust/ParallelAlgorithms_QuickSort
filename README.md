# ParallelAlgorithms_QuickSort

## How to run

1. Установить oneAPI TBB
```
sudo apt install libtbb-dev
```

2. Запустить программу
```
make run
```

3. Результаты будут записаны в `results.txt`

## Results

Sequential #0 timer finished: 47297340 microseconds	/	47297 milliseconds	/	47 seconds

Sequential #1 timer finished: 47117191 microseconds	/	47117 milliseconds	/	47 seconds

Sequential #2 timer finished: 47139671 microseconds	/	47139 milliseconds	/	47 seconds

Sequential #3 timer finished: 47753632 microseconds	/	47753 milliseconds	/	47 seconds

Sequential #4 timer finished: 48179773 microseconds	/	48179 milliseconds	/	48 seconds

Sequential mean time: 4.74975e+07 microseconds

Parallel #0 timer finished: 14352744 microseconds	/	14352 milliseconds	/	14 seconds

Parallel #1 timer finished: 15029275 microseconds	/	15029 milliseconds	/	15 seconds

Parallel #2 timer finished: 17778062 microseconds	/	17778 milliseconds	/	17 seconds

Parallel #3 timer finished: 14045666 microseconds	/	14045 milliseconds	/	14 seconds

Parallel #4 timer finished: 14207424 microseconds	/	14207 milliseconds	/	14 seconds

Parallel mean time: 1.50826e+07 microseconds

**Sequential to Parallel Ratio: 3.14915**
