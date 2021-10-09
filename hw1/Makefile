objects = main.o PLA.o

hw1.out: $(objects)
	clang++ $(objects) -o $@

$(objects): %.o: %.cpp
	clang++ -c $^

clean:
	rm -f *.o *.out