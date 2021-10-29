objects = main.o Regression.o
EIGEN_PATH = /usr/local/include/eigen3

hw2.out: $(objects)
	clang++ $(objects) -o $@

$(objects): %.o: %.cpp
	clang++ -c $^ -I $(EIGEN_PATH)

clean:
	rm -f *.o *.out