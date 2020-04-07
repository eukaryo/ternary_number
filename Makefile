pdep: Source.cpp
	g++ -std=c++1y -O3 -flto -march=native Source.cpp -o ternary
clean:
	rm -f *.o ternary
