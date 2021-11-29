objects = main.o Features.o
EIGEN_PATH = /usr/local/include/eigen3

hw4.out: $(objects)
	clang++ $(objects) -o $@

$(objects): %.o: %.cpp
	clang++ -c $^ -I $(EIGEN_PATH)

clean:
	rm -f *.o *.out