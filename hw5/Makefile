objs = main.o svm_utils.o svm.o

hw5.out: $(objs)
	clang++ $(objs) -o $@

$(objs): %.o: %.cpp
	clang++ -c $^

clean:
	rm -f *.o *.out