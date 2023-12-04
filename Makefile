all:
	g++ main.cpp -ltbb -o main

run: all	
	./main > results.txt

clean:
	rm main results.txt