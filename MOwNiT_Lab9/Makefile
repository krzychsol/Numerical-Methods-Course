CC=gcc -Wall
GSL_HOME=/usr/local/include/gsl

CFLAGS=-I$(GSL_HOME)/include
LDFLAGS=-L$(GSL_HOME)/lib -lgsl -lgslcblas -lm 

all: solution

solution: solution.c
	$(CC) -g -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o