objs = main.o adaboost.o

hw6.out: $(objs)
	clang++ $(objs) -o $@

$(objs): %.o: %.cpp
	clang++ -c $^

clean:
	rm -f *.o *.out